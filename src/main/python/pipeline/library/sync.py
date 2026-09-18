'''Explicit library-sync publishing entry point.'''
from typing import Optional

from pipeline.config import AnalysisConfig
from pipeline.publish.git import RepoTarget
from pipeline.publish.report import ReportSpec
from pipeline.review import publish_reviewed_queue


def sync_library(queue_dir: str, xml_repo: RepoTarget, *, meta_defaults: Optional[dict] = None,
                 images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                 image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                 report_spec: ReportSpec = ReportSpec(), config: AnalysisConfig = AnalysisConfig(),
                 work_dir: Optional[str] = None) -> list[dict]:
    '''Publish accepted review entries; never invoke extraction or design.'''
    return publish_reviewed_queue(
        queue_dir, xml_repo, meta_defaults=meta_defaults, images_repo=images_repo,
        image_owner=image_owner, image_repo_name=image_repo_name, xml_dir=xml_dir, image_dir=image_dir,
        report_spec=report_spec, config=config, work_dir=work_dir,
    )
