"""Command-line interface for tParton PDF evolution.

This module provides the CLI entry point for running tParton from the command line.

Usage
-----
Run with method argument 'm' for Mellin method or 't' for direct integration:

    python -m tparton m    # Vogelsang's Mellin moment method
    python -m tparton t    # Hirai's direct integration method

The specific evolution parameters are defined in the main() functions
within m_evolution.py and t_evolution.py respectively.

See Also
--------
m_evolution : Mellin moment evolution implementation
t_evolution : Direct integration evolution implementation
"""
import sys
method = sys.argv[1]

if method == 'm':
    print("Vogelsang's moment method")
    from .m_evolution import main
    main()
else:
    print("Hirai's energy scale method")
    from .t_evolution import main
    main()