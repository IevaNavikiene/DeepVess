def prepareImage(tif_file, path, tiffile_path):
    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.cd('api/resources/preprocessing/DeepVess',nargout=0)
    eng.prepareImage(tif_file, path, tiffile_path, nargout=0)
    return True

