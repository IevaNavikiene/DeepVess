def prepareImage(path, tiffile_path):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess',nargout=0)
    eng.prepareImage(path, tiffile_path, nargout=0)
    return True

