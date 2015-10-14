#!/usr/bin/env python
# -*- coding: utf8 -*-

import sys
if len(sys.argv)>1: mode = sys.argv[1]
else: mode = 'both'

import elasticite as el

e = el.EdgeGrid(mode=mode)
el.main(e)
