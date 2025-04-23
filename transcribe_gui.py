#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WHYcast Transcribe GUI - Entrypunt voor de grafische interface.

Dit script start de grafische gebruikersinterface van het WHYcast-transcribe programma.
"""

import sys
import os

# Zorg ervoor dat de src directory in het pad staat
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from whycast_transcribe.gui import main

if __name__ == '__main__':
    # Start de GUI
    sys.exit(main())