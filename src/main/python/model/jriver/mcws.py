import logging
from typing import Dict, Optional, Tuple
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

    def as_dict(self) -> dict:
        return {self.__ip: (self.__auth, self.__secure)}

    def __repr__(self):
        suffix = f" [{self.__auth[0]}]" if self.__auth else ' [Unauthenticated]'
        return f"{self.__ip}{suffix}"

    def authenticate(self) -> bool:
        self.__token = None
        url = f"{self.__base_url}/Authenticate"
        r = requests.get(url, auth=self.__auth)
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    for item in response:
                        if item.attrib['Name'] == 'Token':
                            self.__token = item.text
        if self.connected:
            return True
        else:
            raise MCWSError('Authentication failure',  r.url, r.status_code, r.text)

    @property
    def connected(self) -> bool:
        return self.__token is not None

    def get_zones(self) -> Dict[str, str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__base_url}/Playback/Zones", params={'Token': self.__token})
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    zones = {}
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
                    return {v['name']: v['id'] for v in zones.values()}
        raise MCWSError('No zones loaded',  r.url, r.status_code, r.text)

    def __auth_if_required(self):
        if not self.connected:
            self.authenticate()

    def get_dsp(self, zone_id: str) -> Optional[str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__base_url}/Playback/SaveDSPPreset", params={'Token': self.__token, 'Zone': zone_id, 'ZoneType': 'ID'})
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    for child in response:
                        if child.tag == 'Item' and 'Name' in child.attrib and child.attrib['Name'] == 'Preset':
                            return child.text
        raise MCWSError('No DSP loaded',  r.url, r.status_code, r.text)

    def set_dsp(self, zone_name: str, dsp: str) -> bool:
        self.__auth_if_required()
        dsp = dsp.replace('\n', '\r\n')
        if not dsp.endswith('\r\n'):
            dsp = dsp + '\r\n'
        r = requests.post(f"{self.__base_url}/Playback/LoadDSPPreset",
                          params={'Token': self.__token, 'Zone': zone_name, 'ZoneType': 'Name'},
                          files={'Name': (None, dsp)})
        if r.status_code == 200:
            logger.debug(f"LoadDSPPreset/{zone_name} success")
            return True
        else:
            raise MCWSError('DSP not set',  r.url, r.status_code, r.text)


class MCWSError(Exception):

    def __init__(self, msg: str, url: str, status_code: int, resp: Optional[str] = None):
        super().__init__(msg)
        self.msg = msg
        self.url = url
        self.status_code = status_code
        self.resp = resp
