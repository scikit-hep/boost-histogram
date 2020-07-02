# -*- coding: utf-8 -*-
import pytest


@pytest.fixture(params=(False, True), ids=("nogrowth", "growth"))
def growth(request):
    return request.param
