with import <nixpkgs> { };

let
  pythonPackages = python3Packages;
in
pkgs.mkShell rec {
  name = "impurePythonEnv";
  venvDir = "./.venv";
  myPythonPackages = with pythonPackages; [
    # A Python interpreter including the 'venv' module is required to bootstrap
    # the environment.
    python
    # This executes some shell code to initialize a venv in $venvDir before
    # dropping into the shell
    venvShellHook

    # Those are dependencies that we would like to use from nixpkgs, which will
    # add them to PYTHONPATH and thus make them accessible from within the venv.
    pip
    numpy
    pandas
    tensorflow
    typer
    keras
    scikitlearn
    matplotlib
  ];
  buildInputs = [

    myPythonPackages
    # In this particular example, in order to compile any binary extensions they may
    # require, the Python modules listed in the hypothetical requirements.txt need
    # the following packages to be installed locally:
    taglib
    openssl
    git
    libxml2
    libxslt
    libzip
    zlib
  ];

  # Run this command, only after creating the virtual environment
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    # check if there is a requirements.txt file
    if test -f requirements.txt; then
      # install the requirements
      pip install -r requirements.txt
    fi
  '';

  # Now we can execute any commands within the virtual environment.
  # This is optional and can be left out to run pip manually.
  postShellHook = ''
    # allow pip to install wheels
    unset SOURCE_DATE_EPOCH
  '';

}
