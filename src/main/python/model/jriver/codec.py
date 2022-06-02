from __future__ import annotations

import xml.etree.ElementTree as et
from typing import Dict, List, Optional

from model.jriver.common import OutputFormat, OUTPUT_FORMATS


def xpath_to_key_data_value(key_name, data_name):
    '''
    an ET compatible xpath to get the value from a DSP config via the path /Preset/Key/Data/Value for a given key and
    data.
    :param key_name:
    :param data_name:
    :return:
    '''
    return f"./Preset/Key[@Name=\"{key_name}\"]/Data/Name[.=\"{data_name}\"]/../Value"


def get_text_value(root, xpath) -> str:
    return get_element(root, xpath).text


def get_element(root, xpath) -> et.Element:
    matches = root.findall(xpath)
    if matches:
        if len(matches) == 1:
            return matches[0]
        else:
            raise ValueError(f"Multiple matches for {xpath}")
    else:
        raise ValueError(f"No matches for {xpath}")


def get_output_format(config_txt: str, allow_padding: bool) -> OutputFormat:
    '''
    :param config_txt: the dsp config.
    :return: the output format.
    '''
    root = et.fromstring(config_txt)

    def xpath_val(key):
        return get_text_value(root, xpath_to_key_data_value('Audio Settings', key))

    output_channels = int(xpath_val('Output Channels'))
    padding = int(xpath_val('Output Padding Channels'))
    try:
        layout = int(xpath_val('Output Channel Layout'))
    except:
        layout = -1
    if allow_padding and padding > 0:
        if layout == 15:
            template = OUTPUT_FORMATS['THREE_ONE']
            xml_vals = (template.output_channels, padding, 15)
        elif output_channels == 3:
            template = OUTPUT_FORMATS['TWO_ONE']
            xml_vals = (template.input_channels, padding)
        else:
            template = OutputFormat.from_output_channels(output_channels)
            xml_vals = (template.output_channels, padding)
        return OutputFormat(f"{template.display_name} (+{padding})", template.input_channels,
                            template.input_channels + padding, template.lfe_channels, xml_vals,
                            template.paddings[-1] if template.paddings else 0)
    else:
        return get_legacy_output_format(output_channels, padding, layout)


def get_legacy_output_format(output_channels: int, padding: int, layout: int) -> OutputFormat:
    if output_channels == 0:
        return OUTPUT_FORMATS['SOURCE']
    elif output_channels == 1:
        return OUTPUT_FORMATS['MONO']
    elif output_channels == 2 and padding == 0:
        return OUTPUT_FORMATS['STEREO']
    elif output_channels == 3:
        return OUTPUT_FORMATS['TWO_ONE']
    elif output_channels == 4 and layout == 15:
        return OUTPUT_FORMATS['THREE_ONE']
    elif output_channels == 2 and padding == 2:
        return OUTPUT_FORMATS['STEREO_IN_FOUR']
    elif output_channels == 2 and padding == 4:
        return OUTPUT_FORMATS['STEREO_IN_FIVE']
    elif output_channels == 2 and padding == 6:
        return OUTPUT_FORMATS['STEREO_IN_SEVEN']
    elif output_channels == 6 and padding == 2:
        return OUTPUT_FORMATS['FIVE_ONE_IN_SEVEN']
    elif output_channels == 4 and padding == 0:
        return OUTPUT_FORMATS['FOUR']
    elif output_channels == 6 or output_channels == 8:
        return OUTPUT_FORMATS['FIVE_ONE'] if output_channels == 6 else OUTPUT_FORMATS['SEVEN_ONE']
    else:
        excess = output_channels - 8
        if excess < 1:
            raise ValueError(f"Illegal combination [ch: {output_channels}, p: {padding}, l: {layout}")
        return OutputFormat.from_output_channels(output_channels)


def write_dsp_file(root, file_name):
    '''
    :param root: the root element.
    :param file_name: the file to write to.
    '''
    tree = et.ElementTree(root)
    tree.write(file_name, encoding='UTF-8', xml_declaration=True)


def get_peq_block_order(config_txt):
    root = et.fromstring(config_txt)
    peq_blocks = []
    for e in root.findall('./Preset/Key[@Name]/Data/Name[.=\"Enabled\"]/../Value[.="1"]/../..'):
        if e.attrib['Name'] == 'Parametric Equalizer':
            peq_blocks.append(0)
        elif e.attrib['Name'] == 'Parametric Equalizer 2':
            peq_blocks.append(1)
    if not peq_blocks:
        raise ValueError(f"No Enabled Parametric Equalizers found in {config_txt}")
    if len(peq_blocks) > 1:
        order_elem = root.find('./Preset/Key[@Name="DSP Studio"]/Data/Name[.="Plugin Order"]/../Value')
        if order_elem is not None:
            block_order = [token for token in order_elem.text.split(')') if 'Parametric Equalizer' in token]
            if block_order:
                if block_order[0].endswith('Parametric Equalizer'):
                    return [0, 1]
                else:
                    return [1, 0]
    return peq_blocks


class NoFiltersError(ValueError):
    pass


def extract_filters(config_txt: str, key_name: str, allow_empty: bool = False):
    '''
    :param config_txt: the xml text.
    :param key_name: the filter key name.
    :param allow_empty: if true, create the missing filters element if it doesn't exist.
    :return: (root element, filter element)
    '''
    root = et.fromstring(config_txt)
    elements = root.findall(xpath_to_key_data_value(key_name, 'Filters'))
    if elements and len(elements) == 1:
        return root, elements[0]
    if allow_empty:
        parent_element = root.find(f"./Preset/Key[@Name=\"{key_name}\"]")
        data_element = et.Element('Data')
        name_element = et.Element('Name')
        name_element.text = 'Filters'
        data_element.append(name_element)
        value_element = et.Element('Value')
        value_element.text = ''
        data_element.append(value_element)
        parent_element.append(data_element)
        return root, value_element
    else:
        raise NoFiltersError(f"No Filters in {key_name} found in {config_txt}")


def get_peq_key_name(block):
    '''
    :param block: 0 or 1.
    :return: the PEQ key name.
    '''
    if block == 0:
        return 'Parametric Equalizer'
    elif block == 1:
        return 'Parametric Equalizer 2'
    else:
        raise ValueError(f"Unknown PEQ block {block}")


def filts_to_xml(vals: List[Dict[str, str]]) -> str:
    '''
    Formats key-value pairs into a jriver dsp config file compatible str fragment.
    :param vals: the key-value pairs.
    :return: the txt snippet.
    '''
    return ''.join(filt_to_xml(f) for f in vals)


def filt_to_xml(vals: Dict[str, str]) -> str:
    '''
    Converts a set of filter values to a jriver compatible xml fragment.
    :param vals: the values.
    :return: the xml fragment.
    '''
    items = [f"<Item Name=\"{k}\">{v}</Item>" for k, v in vals.items()]
    catted_items = '\n'.join(items)
    prefix = '<XMLPH version="1.1">'
    suffix = '</XMLPH>'
    txt_length = len(prefix) + len(''.join(items)) + len(suffix)
    new_line_len = (len(items) + 1) * 2
    total_len = txt_length + new_line_len
    xml_frag = f"({total_len}:{prefix}\n{catted_items}\n{suffix})"
    # print(f"{filter_classes_by_type[vals['Type']].__name__} ({vals['Type']}): {offset}")
    return xml_frag


def include_filters_in_dsp(peq_block_name: str, config_txt: str, xml_filts: List[str], replace: bool = True) -> str:
    '''
    :param peq_block_name: the peq block to process.
    :param config_txt: the dsp config in txt form.
    :param xml_filts: the filters to include.
    :param replace: if true, replace existing filters. if false, append.
    :return: the new config txt.
    '''
    if xml_filts:
        root, filt_element = extract_filters(config_txt, peq_block_name, allow_empty=True)
        # before_value, after_value, filt_section = extract_value_section(config_txt, self.__block)
        # separate the tokens, which are in (TOKEN) blocks, from within the Value element
        if filt_element.text:
            filt_fragments = [v + ')' for v in filt_element.text.split(')') if v]
            if len(filt_fragments) < 2:
                raise ValueError('Invalid input file - Unexpected <Value> format')
        else:
            filt_fragments = ['(1:1)', '(2:0)']
        # find the filter count and replace it with the new filter count
        new_filt_count = sum([x.count('<XMLPH version') for x in xml_filts])
        if not replace:
            new_filt_count = int(filt_fragments[1][1:-1].split(':')[1]) + new_filt_count
        filt_fragments[1] = f"({len(str(new_filt_count))}:{new_filt_count})"
        # append the new filters to any existing ones or replace
        if replace:
            new_filt_section = ''.join(filt_fragments[0:2]) + ''.join(xml_filts)
        else:
            new_filt_section = ''.join(filt_fragments) + ''.join(xml_filts)
        # replace the value block in the original string
        filt_element.text = new_filt_section
        config_txt = et.tostring(root, encoding='UTF-8', xml_declaration=True).decode('utf-8')
        return config_txt
    else:
        return config_txt


def item_to_dicts(frag) -> Optional[Dict[str, str]]:
    idx = frag.find(':')
    if idx > -1:
        peq_xml = frag[idx+1:-1]
        vals = {i.attrib['Name']: i.text for i in et.fromstring(peq_xml).findall('./Item')}
        if 'Enabled' in vals:
            if vals['Enabled'] != '0' and vals['Enabled'] != '1':
                vals['Enabled'] = '1'
        else:
            vals['Enabled'] = '0'
        return vals
    return None
