"""Shared fixtures. Run the tests from the repository root:  python -m pytest -q"""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import build_atlas  # noqa: E402


def project_config(name):
    with open(os.path.join(ROOT, "projects", name, "config.json"), encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def B():
    """The build_atlas module."""
    return build_atlas


@pytest.fixture
def proj():
    """The CAMELS project settings."""
    return build_atlas.Project(project_config("camels"))


@pytest.fixture
def cfg():
    """A fresh copy of the CAMELS config, to modify in a test."""
    return copy.deepcopy(project_config("camels"))


def record(uid, title, **kw):
    """A paper record as build_data makes it, before the overrides."""
    r = {"uid": uid, "title": title, "abstract": "An abstract.", "date": "2024-01-02", "authors": ["Ana Lopez"],
         "link": f"arXiv:{uid}", "url": f"https://arxiv.org/abs/{uid}", "dq": False, "excerpt": False}
    r.update(kw)
    return r
