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