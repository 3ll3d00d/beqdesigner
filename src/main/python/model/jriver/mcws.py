import logging
from typing import Dict, Optional
from xml.etree import ElementTree as ET

import requests

logger = logging.getLogger('jriver.mcws')


class MediaServer:

    def __init__(self, ip: str, user: str, password: str, secure: bool = False):
        self.__ip = ip
        self.__user = user
        self.__password = password
        self.__secure = secure
        self.__url = f"http{'s' if secure else ''}://{ip}/MCWS/v1"
        self.__token = None

    def as_dict(self) -> dict:
        return {self.__ip: (self.__user, self.__password, self.__secure)}

    def __repr__(self):
        return f"{self.__ip} [{self.__user}]"

    def authenticate(self) -> bool:
        self.__token = None
        r = requests.get(f"{self.__url}/Authenticate", auth=(self.__user, self.__password))
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    for item in response:
                        if item.attrib['Name'] == 'Token':
                            self.__token = item.text
        return self.connected

    @property
    def connected(self) -> bool:
        return self.__token is not None

    def get_zones(self) -> Dict[str, str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__url}/Playback/Zones", params={'Token': self.__token})
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
        return {}

    def __auth_if_required(self):
        if not self.connected:
            self.authenticate()

    def get_dsp(self, zone_name: str) -> Optional[str]:
        self.__auth_if_required()
        r = requests.get(f"{self.__url}/Playback/SaveDSPPreset", params={'Token': self.__token, 'Zone': zone_name, 'ZoneType': 'Name'})
        if r.status_code == 200:
            response = ET.fromstring(r.content)
            if response:
                r_status = response.attrib.get('Status', None)
                if r_status == 'OK':
                    for child in response:
                        if child.tag == 'Item' and 'Name' in child.attrib and child.attrib['Name'] == 'Preset':
                            return child.text
        return None

    def set_dsp(self, zone_name: str, dsp: str) -> bool:
        self.__auth_if_required()
        dsp = dsp.replace('\n', '\r\n')
        if not dsp.endswith('\r\n'):
            dsp = dsp + '\r\n'
        with open('/tmp/test.dsp', mode='w') as f:
            f.write(dsp)
        # r = requests.get(f"{self.__url}/Playback/LoadDSPPreset",
        #                  params={'Token': self.__token, 'Zone': zone_name, 'ZoneType': 'Name', 'Name': dsp})
        r = requests.post(f"{self.__url}/Playback/LoadDSPPreset",
                          params={'Token': self.__token, 'Zone': zone_name, 'ZoneType': 'Name'},
                          files={'Name': (None, dsp)})
        if r.status_code == 200:
            logger.debug(f"LoadDSPPreset/{zone_name} success")
            return True
        else:
            logger.warning(f"LoadDSPPreset/{zone_name} received {r.status_code} {r.text}")
            return False
