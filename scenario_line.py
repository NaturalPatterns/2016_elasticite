#!/usr/bin/env python
# -*- coding: utf8 -*-

import elasticite as el

e = el.EdgeGrid(N_lame=20, grid_type='line')
el.main(e)
