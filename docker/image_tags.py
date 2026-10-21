'''
The tags a release's pipeline image is pushed under (create-image.yaml):

    python3 docker/image_tags.py ghcr.io/3ll3d00d/beqdesigner-pipeline 2.2.0-alpha.23

prints them comma-separated. Every release gets its own tag and `alpha`, which follows the newest release of any kind,
so a container pointed at `:alpha` is moved on by a plain pull and recreate (Unraid's update). `latest` follows releases
without a pre-release suffix only.
'''
import re
import sys
from typing import List

PRERELEASE = re.compile(r'-(alpha|beta|rc)([.-]|$)')


def image_tags(image: str, ref_name: str) -> List[str]:
    tags = [f'{image}:{ref_name}', f'{image}:alpha']
    if not PRERELEASE.search(ref_name):
        tags.append(f'{image}:latest')
    return tags


if __name__ == '__main__':
    print(','.join(image_tags(sys.argv[1], sys.argv[2])))
