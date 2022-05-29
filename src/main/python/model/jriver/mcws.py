import logging
from typing import Dict, Optional, Tuple, Callable
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger('jriver.mcws')


class MediaServer:

    def __init__(self, ip: str, auth: Optional[Tuple[str, str]] = None, secure: bool = False):
        self.__ip = ip
        self.__auth = auth
        self.__secure = secure
        self.__base_url = f"http{'s' if secure else ''}://{ip}/MCWS/v1"
        self.__token = None
        self.__major_version = None

    def as_dict(self) -> dict:
        return {self.__ip: (self.__auth, self.__secure)}

    def __repr__(self):
        suffix = f" [{self.__auth[0]}]" if self.__auth else ' [Unauthenticated]'
        return f"{self.__ip}{suffix}"

    def authenticate(self) -> bool:
        self.__token = None
        url = f"{self.__base_url}/Authenticate"
        r = requests.get(url, auth=self.__auth, timeout=(1, 5))
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    for item in response:
                        if item.attrib['Name'] == 'Token':
                            self.__token = item.text
        if self.connected:
            self.__load_version()
            return True
        else:
            raise MCWSError('Authentication failure', r.url, r.status_code, r.text)

    def __load_version(self):
        url = f"{self.__base_url}/Alive"
        r = requests.get(url, params={'Token': self.__token}, timeout=(1, 5))
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    v = next((item.text for item in response if item.attrib['Name'] == 'ProgramVersion'), None)
                    if v:
                        tokens = str(v).split('.')
                        if tokens:
                            try:
                                self.__major_version = int(tokens[0])
                            except:
                                raise MCWSError(f"Unknown version format {v}", r.url, r.status_code, r.text)
        if not self.__major_version:
            raise MCWSError('No version', r.url, r.status_code, r.text)

    @property
    def connected(self) -> bool:
        return self.__token is not None

    def get_zones(self) -> Dict[str, str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__base_url}/Playback/Zones", params={'Token': self.__token}, timeout=(1, 5))
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    zones = {}
                    remote_zones = []
                    for child in response:
                        if child.tag == 'Item' and 'Name' in child.attrib:
                            attrib = child.attrib['Name']
                            if attrib.startswith('ZoneName'):
                                item_idx = attrib[8:]
                                if item_idx in zones:
                                    zones[item_idx]['name'] = child.text
                                else:
                                    zones[item_idx] = {'name': child.text}
                            elif attrib.startswith('ZoneID'):
                                item_idx = attrib[6:]
                                if item_idx in zones:
                                    zones[item_idx]['id'] = child.text
                                else:
                                    zones[item_idx] = {'id': child.text}
                            elif attrib.startswith('ZoneDLNA'):
                                if child.text == '1':
                                    remote_zones.append(attrib[8:])
                    return {(v['name'] if k not in remote_zones else f"{v['name']} (DLNA)"): v['id'] for k, v in
                            zones.items() if k not in remote_zones}
        raise MCWSError('No zones loaded', r.url, r.status_code, r.text)

    def __auth_if_required(self):
        if not self.connected:
            self.authenticate()

    def get_dsp(self, zone_id: str) -> Optional[str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__base_url}/Playback/SaveDSPPreset",
                         params={'Token': self.__token, 'Zone': zone_id, 'ZoneType': 'ID'},
                         timeout=(1, 5))
        if r.status_code == 200:
            response = ET.fromstring(r.text)
            if response:
                if response.tag == 'DSP':
                    return r.text
                elif response.tag == 'Response':
                    r_status = response.attrib.get('Status', None)
                    if r_status == 'OK':
                        for child in response:
                            if child.tag == 'Item' and 'Name' in child.attrib and child.attrib['Name'] == 'Preset':
                                return child.text
        raise MCWSError('No DSP loaded', r.url, r.status_code, r.text)

    def set_dsp(self, zone_id: str, dsp_txt_provider: Callable[[bool], str]) -> bool:
        self.__auth_if_required()
        dsp = dsp_txt_provider(self.convert_q)
        dsp = dsp.replace('\n', '\r\n')
        if not dsp.endswith('\r\n'):
            dsp = dsp + '\r\n'
        r = requests.post(f"{self.__base_url}/Playback/LoadDSPPreset",
                          params={'Token': self.__token, 'Zone': zone_id, 'ZoneType': 'ID'},
                          files={'Name': (None, dsp)},
                          timeout=(1, 5))
        if r.status_code == 200:
            logger.debug(f"LoadDSPPreset/{zone_id} success")
            loaded_dsp = self.get_dsp(zone_id)
            if self.__compare_xml(ET.fromstring(dsp), ET.fromstring(loaded_dsp)):
                return True
            else:
                raise DSPMismatchError(zone_id, dsp, loaded_dsp)
        else:
            raise MCWSError('DSP not set', r.url, r.status_code, r.text)

    @property
    def mc_version(self) -> Optional[int]:
        return self.__major_version

    @property
    def can_pad_output_channels(self) -> bool:
        return self.__is_29()

    @property
    def convert_q(self) -> bool:
        return not self.__is_29()

    def __is_29(self) -> bool:
        return True if self.mc_version and self.mc_version >= 29 else False

    @staticmethod
    def __compare_xml(x1, x2):
        if x1.tag != x2.tag:
            return False
        for name, value in x1.attrib.items():
            if x2.attrib.get(name) != value:
                return False
        for name in x2.attrib:
            if name not in x1.attrib:
                return False
        if not MediaServer.__text_compare(x1.text, x2.text):
            return False
        if not MediaServer.__text_compare(x1.tail, x2.tail):
            return False
        cl1 = list(x1)
        cl2 = list(x2)
        if len(cl1) != len(cl2):
            return False
        i = 0
        for c1, c2 in zip(cl1, cl2):
            i += 1
            if not MediaServer.__compare_xml(c1, c2):
                return False
        return True

    @staticmethod
    def __text_compare(t1, t2):
        if not t1 and not t2:
            return True
        if t1 == '*' or t2 == '*':
            return True
        return (t1 or '').strip() == (t2 or '').strip()


class DSPMismatchError(Exception):

    def __init__(self, zone_id: str, expected: str, actual):
        super().__init__(f"Mismatch in DSP loaded to {zone_id}")
        self.zone_id = zone_id
        self.expected = expected
        self.actual = actual


class MCWSError(Exception):

    def __init__(self, msg: str, url: str, status_code: int, resp: Optional[str] = None):
        super().__init__(msg)
        self.msg = msg
        self.url = url
        self.status_code = status_code
        self.resp = resp
