import logging
import os
from abc import abstractmethod, ABC
from uuid import uuid4

from qtpy.QtCore import QObject, Signal, QRunnable
from qtpy.QtWidgets import QFileDialog

from model.iir import Passthrough, PeakingEQ, Shelf, LowShelf, HighShelf
from model.preferences import BEQ_DOWNLOAD_DIR

logger = logging.getLogger('minidsp')

BMILLER_GITHUB_MINIDSP = 'https://github.com/bmiller/miniDSPBEQ'
BMILLER_MINI_DSPBEQ_GIT_REPO = f"{BMILLER_GITHUB_MINIDSP}.git"


class XmlParser(ABC):
    def __init__(self, minidsp_type, optimise_filters):
        self.__minidsp_type = minidsp_type
        self.__optimise_filters = optimise_filters

    @property
    def minidsp_type(self):
        return self.__minidsp_type

    def __pad(self, filt):
        was_optimised = False
        try:
            filters = pad_with_passthrough(filt,
                                           fs=self.minidsp_type.target_fs,
                                           required=self.minidsp_type.filters_required,
                                           optimise=self.__optimise_filters)
        except OptimisedFilters as e:
            was_optimised = True
            filters = e.flattened_filters
        return filters, was_optimised

    def convert(self, dst, filt, metadata=None):
        filters, was_optimised = self.__pad(filt)
        output_config = self._overwrite(filters, dst, metadata)
        return output_config, was_optimised

    def file_extension(self):
        return '.xml'

    @abstractmethod
    def _overwrite(self, filters, target, metadata=None):
        pass


class TwoByFourXmlParser(XmlParser):
    '''
    Handles the 2x4 model
    '''
    def __init__(self, minidsp_type, optimise_filters):
        super().__init__(minidsp_type, optimise_filters)

    def _overwrite(self, filters, target, metadata=None):
        import xml.etree.ElementTree as ET
        import re
        logger.info(f"Copying {len(filters)} to {target}")
        et_tree = ET.parse(str(target))
        root = et_tree.getroot()
        filter_matcher = re.compile('^EQ_ch([1-2])_1_([1-6])$')
        bq_matcher = re.compile('^EQ_ch([1-2])_1_([1-6])_([A-B][0-2])$')
        for child in root:
            if child.tag == 'filter':
                filter_name = child.attrib['name']
                matches = filter_matcher.match(filter_name)
                if matches is not None and len(matches.groups()) == 2:
                    filt_slot = matches.group(2)
                    if int(filt_slot) > len(filters):
                        root.remove(child)
                    else:
                        filt = filters[int(filt_slot) - 1]
                        if isinstance(filt, Passthrough):
                            child.find('freq').text = '1000'
                            child.find('q').text = '1'
                            child.find('gain').text = '0'
                            child.find('boost').text = '0'
                            child.find('type').text = 'PK'
                            child.find('bypass').text = '1'
                            child.find('basic').text = 'true'
                        else:
                            child.find('freq').text = str(filt.freq)
                            child.find('q').text = str(filt.q)
                            child.find('boost').text = str(filt.gain)
                            child.find('type').text = get_minidsp_filter_code(filt)
                            child.find('bypass').text = '0'
                            child.find('basic').text = 'true'
            elif child.tag == 'item':
                filter_name = child.attrib['name']
                matches = bq_matcher.match(filter_name)
                if matches is not None and len(matches.groups()) == 3:
                    filt_slot = matches.group(2)
                    biquad_coeff = matches.group(3)
                    if int(filt_slot) > len(filters):
                        root.remove(child)
                    else:
                        filt = filters[int(filt_slot) - 1]
                        if isinstance(filt, Passthrough):
                            child.find('dec').text = '0'
                            child.find('hex').text = '00800000' if biquad_coeff == 'B0' else '00800000'
                        else:
                            child.find('dec').text = '0'
                            hex_txt = filt.format_biquads(True, separator=',', show_index=True, to_hex=True,
                                                          fixed_point=True)[0]
                            hex_val = dict(item.split("=") for item in hex_txt.split(','))[biquad_coeff.lower()]
                            child.find('hex').text = hex_val

        return ET.tostring(root, encoding='unicode')


class HDXmlParser(XmlParser):
    '''
    Handles HD models (2x4HD and 10x10HD)
    '''
    def __init__(self, minidsp_type, optimise_filters, selected_channels):
        super().__init__(minidsp_type, optimise_filters)
        if selected_channels:
            self.__selected_channels = [self.__extract_channel(i) for i in selected_channels]
        else:
            self.__selected_channels = minidsp_type.filter_channels

    @staticmethod
    def __extract_channel(txt):
        if len(txt) == 1:
            return txt[0]
        elif txt[0:5] == 'Input':
            return txt[-1]
        elif txt[0:6] == 'Output':
            return str(int(txt[-1]) + 2)
        else:
            raise ValueError(f"Unsupported channel {txt}")

    def _overwrite(self, filters, target, metadata=None):
        '''
        Overwrites the PEQ_1_x and PEQ_2_x filters (or the 1-4 filters for the SHD).
        :param filters: the filters.
        :param target: the target file.
        :param metadata: the minidsp metadata.
        :return: the xml to output to the file.
        '''
        import xml.etree.ElementTree as ET
        logger.info(f"Copying {len(filters)} to {target}")
        et_tree = ET.parse(str(target))
        root = et_tree.getroot()
        for child in root:
            if child.tag == 'filter':
                if 'name' in child.attrib:
                    filter_tokens = child.attrib['name'].split('_')
                    (filt_type, filt_channel, filt_slot) = filter_tokens
                    if len(filter_tokens) == 3:
                        if filt_type == 'PEQ':
                            if filt_channel in self.__selected_channels:
                                if int(filt_slot) > len(filters):
                                    root.remove(child)
                                else:
                                    filt = filters[int(filt_slot)-1]
                                    if isinstance(filt, Passthrough):
                                        child.find('freq').text = '1000'
                                        child.find('q').text = '0.7'
                                        child.find('boost').text = '0'
                                        child.find('type').text = 'PK'
                                        child.find('bypass').text = '1'
                                    else:
                                        child.find('freq').text = str(filt.freq)
                                        child.find('q').text = str(filt.q)
                                        child.find('boost').text = str(filt.gain)
                                        child.find('type').text = get_minidsp_filter_code(filt)
                                        child.find('bypass').text = '0'
                                    dec_txt = filt.format_biquads(True, separator=',',
                                                                  show_index=False, to_hex=False)[0]
                                    child.find('dec').text = f"{dec_txt},"
                                    hex_txt = filt.format_biquads(True, separator=',',
                                                                  show_index=False, to_hex=True,
                                                                  fixed_point=self.minidsp_type.is_fixed_point_hardware())[0]
                                    child.find('hex').text = f"{hex_txt},"
        if metadata is not None:
            metadata_tag = ET.Element('beq_metadata')
            for key, value in metadata.items():
                tag = ET.Element(key)

                if type(value) is list:
                    for item in value:
                        sub_tag = ET.Element('value')
                        sub_tag.text = item
                        tag.append(sub_tag)
                else:
                    tag.text = metadata[key]
                metadata_tag.append(tag)

            root.append(metadata_tag)

        return ET.tostring(root, encoding='unicode')


def get_minidsp_filter_code(filt):
    '''
    :param filt: the filter.
    :return: the string filter type for a minidsp xml.
    '''
    if isinstance(filt, PeakingEQ):
        return 'PK'
    elif isinstance(filt, LowShelf):
        return 'SL'
    elif isinstance(filt, HighShelf):
        return 'SH'
    else:
        raise ValueError(f"Unknown minidsp filter type {type(filt)}")


def xml_to_filt(file, fs=1000, unroll=False):
    ''' Extracts a set of filters from the provided minidsp file '''
    from model.iir import PeakingEQ, LowShelf, HighShelf

    filts = __extract_filters(file)
    output = []
    for filt_tup, count in filts.items():
        filt_dict = dict(filt_tup)
        if filt_dict['type'] == 'SL':
            for i in range(0, count if unroll is True else 1):
                filt = LowShelf(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                                count=1 if unroll is True else count)
                output.append(filt)
        elif filt_dict['type'] == 'SH':
            for i in range(0, count if unroll is True else 1):
                filt = HighShelf(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']),
                                 count=1 if unroll is True else count)
                output.append(filt)
        elif filt_dict['type'] == 'PK':
            for i in range(0, count):
                filt = PeakingEQ(fs, float(filt_dict['freq']), float(filt_dict['q']), float(filt_dict['boost']))
                output.append(filt)
        else:
            logger.info(f"Ignoring unknown filter type {filt_dict}")
    return output


def __extract_filters(file):
    import xml.etree.ElementTree as ET
    from collections import Counter

    ignore_vals = ['hex', 'dec']
    tree = ET.parse(file)
    root = tree.getroot()
    filts = {}
    for child in root:
        if child.tag == 'filter':
            if 'name' in child.attrib:
                current_filt = None
                filter_tokens = child.attrib['name'].split('_')
                (filt_type, filt_channel, filt_slot) = filter_tokens
                if len(filter_tokens) == 3:
                    if filt_type == 'PEQ':
                        if filt_channel not in filts:
                            filts[filt_channel] = {}
                        filt = filts[filt_channel]
                        if filt_slot not in filt:
                            filt[filt_slot] = {}
                        current_filt = filt[filt_slot]
                        for val in child:
                            if val.tag not in ignore_vals:
                                current_filt[val.tag] = val.text
                if current_filt is not None:
                    if 'bypass' in current_filt and current_filt['bypass'] == '1':
                        del filts[filt_channel][filt_slot]
                    elif 'boost' in current_filt and current_filt['boost'] == '0':
                        del filts[filt_channel][filt_slot]
    final_filt = None
    # if 1 and 2 are identical then throw one away
    if '1' in filts and '2' in filts:
        filt_1 = filts['1']
        filt_2 = filts['2']
        if filt_1 == filt_2:
            final_filt = list(filt_1.values())
        else:
            raise ValueError(f"Different input filters found in {file} - Input 1: {filt_1} - Input 2: {filt_2}")
    elif '1' in filts:
        final_filt = list(filts['1'].values())
    elif '2' in filts:
        final_filt = list(filts['2'].values())
    else:
        if len(filts.keys()) == 1:
            for k in filts.keys():
                final_filt = filts[k]
        else:
            raise ValueError(f"Multiple active filters found in {file} - {filts}")
    if final_filt is None:
        raise ValueError(f"No filters found in {file}")
    return Counter([tuple(f.items()) for f in final_filt])


def extract_and_pad_with_passthrough(filt_xml, fs, required=10, optimise=False):
    '''
    Extracts the filters from the XML and pads with passthrough filters.
    :param filt_xml: the xml file.
    :param fs: the target fs.
    :param required: how many filters do we need.
    :param optimise: whether to optimise filters that use more than required biquads.
    :return: the filters.
    '''
    return pad_with_passthrough(xml_to_filt(filt_xml, fs=fs), fs, required, optimise=optimise)


def pad_with_passthrough(filters, fs, required, optimise=False):
    '''
    Pads to the required number of biquads. If the filter uses more than required and optimise is true, attempts to
    squeeze the biquad count.
    :param filters: the filters.
    :param fs: sample rate.
    :param required: no of required biquads.
    :param optimise: whether to try to reduce biquad count.
    :return: the raw biquad filters.
    '''
    flattened_filters = flatten_filters(filters)
    padding = required - len(flattened_filters)
    if padding > 0:
        pad_filters = [Passthrough(fs=fs)] * padding
        flattened_filters.extend(pad_filters)
    elif padding < 0:
        if optimise is True:
            from model.filter import optimise_filters
            padded = pad_with_passthrough(optimise_filters(filters, fs, -padding), fs, required, optimise=False)
            raise OptimisedFilters(padded)
        raise TooManyFilters(f"BEQ has too many filters for device (remove {abs(padding)} biquads)")
    return flattened_filters


def flatten_filters(filter):
    '''
    Flattens the provided filter, i.e. unrolls shelf filters.
    :param filter: the filter.
    :return: the flattened filters as a list.
    '''
    flattened_filters = []
    for filt in filter:
        if isinstance(filt, PeakingEQ):
            flattened_filters.append(filt)
        elif isinstance(filt, Shelf):
            flattened_filters.extend(filt.flatten())
    return flattened_filters


class TooManyFilters(Exception):
    pass


class OptimisedFilters(Exception):
    def __init__(self, flattened_filters):
        self.flattened_filters = flattened_filters


class RefreshSignals(QObject):
    on_start = Signal()
    on_end = Signal()


class RepoRefresher(QRunnable):

    def __init__(self, repo_dir, repos):
        super().__init__()
        self.repo_dir = repo_dir
        self.repos = repos
        self.signals = RefreshSignals()

    def run(self):
        self.signals.on_start.emit()
        try:
            self.refresh()
        except:
            pass
        self.signals.on_end.emit()

    def refresh(self):
        ''' Pulls or clones the named repository '''
        from app import wait_cursor
        with wait_cursor():
            os.makedirs(self.repo_dir, exist_ok=True)
            for repo in self.repos:
                subdir = get_repo_subdir(repo)
                git_metadata_dir = os.path.abspath(os.path.join(self.repo_dir, subdir, '.git'))
                local_dir = os.path.join(self.repo_dir, subdir)
                if os.path.exists(git_metadata_dir):
                    from dulwich.errors import NotGitRepository
                    try:
                        self.__pull_beq(repo, local_dir)
                    except NotGitRepository as e:
                        logger.exception('.git exists but is not a git repo, attempting to delete .git directory and clone')
                        os.rmdir(git_metadata_dir)
                        self.__clone_beq(repo, local_dir)
                else:
                    self.__clone_beq(repo, local_dir)

    @staticmethod
    def __pull_beq(repo, local_dir):
        ''' pulls the git repo but does not use dulwich pull as it has file lock issues on windows '''
        from dulwich import porcelain, index
        with porcelain.open_repo_closing(local_dir) as local_repo:
            remote_refs = porcelain.fetch(local_repo, repo)
            local_repo[b"HEAD"] = remote_refs[b"refs/heads/master"]
            index_file = local_repo.index_path()
            tree = local_repo[b"HEAD"].tree
            index.build_index_from_tree(local_repo.path, index_file, local_repo.object_store, tree)

    @staticmethod
    def __clone_beq(repo, local_dir):
        ''' clones the git repo into the local dir. '''
        from dulwich import porcelain
        porcelain.clone(repo, local_dir, checkout=True)


def get_repo_subdir(repo):
    '''
    Extracts the local subdir from the repo url
    :param repo: the repo url.
    :return the subdir.
    '''
    chunks = repo.split('/')
    if len(chunks) > 1:
        subdir = f"{chunks[-2]}_{chunks[-1]}"
    else:
        subdir = chunks[-1]
    return subdir[0:-4] if subdir[-4:] == '.git' else subdir


def get_commit_url(repo):
    '''
    Extracts the commit url from the repo url
    :param repo: the repo url.
    :return the commit url.
    '''
    return f"{repo[0:-4]}/commit/"


def load_as_filter(parent, preferences, fs, unroll=False):
    '''
    Load a minidsp xml file as a filter.
    '''
    selected = QFileDialog.getOpenFileName(parent=parent, directory=preferences.get(BEQ_DOWNLOAD_DIR),
                                           caption='Load Minidsp XML Filter', filter='Filter (*.xml)')
    filt_file = selected[0] if selected is not None else None
    if filt_file is not None and len(filt_file) > 0:
        filt = xml_to_filt(filt_file, fs, unroll=unroll)
        if filt is not None and len(filt) > 0:
            for f in filt:
                f.id = uuid4()
            return filt, filt_file
    return None
