def prepareImage(stack_path, z_start):
    import matlab.engine
    import os
    eng = matlab.engine.start_matlab()
    eng.prepareImage(z_start, 1, 4, 4, stack_path, nargout=0)
    path, h5file_path = os.path.split(os.path.abspath(stack_path))
    return path, h5file_path