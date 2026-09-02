'''
to_beq_xml(): a thin, Qt-free wrapper over the existing HDXmlParser/
flat24hd.xml path (model/minidsp.py) -- design/pipeline-implementation-plan.md
phase 2 (item 2c). This side of the publish contract already worked
headlessly before this pipeline existed (design/api-headless-pipeline.md
headline finding 1); this module exists to fix the template-path resolution
and to give it a stable, non-GUI entry point.

model/postbuilder.py's own template-path resolution
(`os.path.dirname('__file__')`, a quoted string literal, not the __file__
variable) resolves relative to the current working directory rather than
this repo's layout -- works by accident only when the process happens to be
launched from a particular directory. Fixed properly here.
'''
import os
import sys

from model.merge import DspType
from model.minidsp import HDXmlParser

from pipeline.metadata import BeqMetadata, validate


def flat24hd_template_path() -> str:
    ''' :return: the path to the bundled flat24hd.xml template. '''
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'flat24hd.xml')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'xml', 'flat24hd.xml'))


def to_beq_xml(filters, meta: BeqMetadata, dsp_type=DspType.MINIDSP_TWO_BY_FOUR_HD, pretty: bool = True) -> str:
    '''
    :param filters: an iterable of Biquad (e.g. a CompleteFilter from
        pipeline.designer.convert.to_complete_filter or pipeline.filters.create_filter).
    :param meta: the metadata to embed.
    :param dsp_type: the target device type -- defaults to the 2x4HD this
        pipeline is scoped to (design/api-headless-pipeline.md's out-of-scope note).
    :param pretty: pretty-print the XML.
    :return: the beqcatalogue-format XML.
    :raises ValueError: if meta fails pipeline.metadata.validate(), or if
        filters contains something unpublishable (a Gain filter, an unknown
        type, or too many biquads) -- raised by the existing HDXmlParser/
        flatten_filters/pad_with_passthrough chain this wraps.
    '''
    problems = validate(meta)
    if problems:
        raise ValueError(f"Invalid metadata: {'; '.join(problems)}")

    template = flat24hd_template_path()
    output_xml, _ = HDXmlParser(dsp_type, False).convert(template, filters, meta.to_dict(), pretty=pretty)
    return output_xml
