'''Explicit library-sync publishing entry points: publish (write), commit (commit + push), and both.'''
from typing import Optional

from pipeline.config import AnalysisConfig
from pipeline.library.commit import CatalogueCommit, commit_catalogue
from pipeline.publish.git import RepoTarget
from pipeline.publish.report import ReportSpec
from pipeline.review import publish_reviewed_queue, split_publish_results


def publish_library(queue_dir: str, xml_repo: RepoTarget, *, meta_defaults: Optional[dict] = None,
                    images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                    image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                    report_spec: ReportSpec = ReportSpec(), config: AnalysisConfig = AnalysisConfig(),
                    work_dir: Optional[str] = None) -> list[dict]:
    '''
    Writes every accepted entry's XML (and report image) into the repos' working trees and marks it 'published'.
    Nothing is committed or pushed -- that is commit_library(). Never invokes extraction or design.
    '''
    return publish_reviewed_queue(
        queue_dir, xml_repo, meta_defaults=meta_defaults, images_repo=images_repo,
        image_owner=image_owner, image_repo_name=image_repo_name, xml_dir=xml_dir, image_dir=image_dir,
        report_spec=report_spec, config=config, work_dir=work_dir, push=False,
    )


def commit_library(queue_dir: str, xml_repo: RepoTarget, *, images_repo: Optional[RepoTarget] = None,
                   xml_dir: str = '', image_dir: str = '', push: bool = True) -> CatalogueCommit:
    ''' Commits and pushes what publish_library() wrote: one commit and one push per repo, images first. '''
    return commit_catalogue(queue_dir, xml_repo, images_repo, xml_dir=xml_dir, image_dir=image_dir, push=push)


def sync_library(queue_dir: str, xml_repo: RepoTarget, *, meta_defaults: Optional[dict] = None,
                 images_repo: Optional[RepoTarget] = None, image_owner: Optional[str] = None,
                 image_repo_name: Optional[str] = None, xml_dir: str = '', image_dir: str = '',
                 report_spec: ReportSpec = ReportSpec(), config: AnalysisConfig = AnalysisConfig(),
                 work_dir: Optional[str] = None, push: bool = True) -> list[dict]:
    '''
    publish_library() followed by commit_library(), so a batch is one commit and one push per repo.

    :return: publish_library()'s results, each published one also carrying the batch's `xml_commit` (and
        `image_commit`) sha where a commit was made. A file already committed by an earlier, interrupted run gets none.
    '''
    results = publish_library(
        queue_dir, xml_repo, meta_defaults=meta_defaults, images_repo=images_repo, image_owner=image_owner,
        image_repo_name=image_repo_name, xml_dir=xml_dir, image_dir=image_dir, report_spec=report_spec,
        config=config, work_dir=work_dir)
    committed = commit_library(queue_dir, xml_repo, images_repo=images_repo, xml_dir=xml_dir, image_dir=image_dir,
                               push=push)
    published, _ = split_publish_results(results)
    for result in published:
        if committed.xml.commit:
            result['xml_commit'] = committed.xml.commit
        if committed.images is not None and committed.images.commit:
            result['image_commit'] = committed.images.commit
    return results
