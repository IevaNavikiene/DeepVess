def prepareImage(path, tiffile_path, z_start):
    import matlab.engine
    import os
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess',nargout=0)
    eng.prepareImage(z_start, 4., 4., path, tiffile_path, nargout=0)
    return True

