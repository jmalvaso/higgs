#!/usr/bin/env bash

setup_higgs_cp() {
    # Runs the project setup, leading to a collection of environment variables starting with either
    #   - "CF_", for controlling behavior implemented by columnflow, or
    #   - "HIGGS_CP_", for features provided by the analysis repository itself.
    # Check the setup.sh in columnflow for documentation of the "CF_" variables. The purpose of all
    # "HIGGS_CP_" variables is documented below.
    #
    # The setup also handles the installation of the software stack via virtual environments, and
    # optionally an interactive setup where the user can configure certain variables.
    #
    #
    # Arguments:
    #   1. The name of the setup. "default" (which is itself the default when no name is set)
    #      triggers a setup with good defaults, avoiding all queries to the user and the writing of
    #      a custom setup file. See "interactive_setup()" for more info.
    #
    #
    # Optinally preconfigured environment variables:
    #   None yet.
    #
    #
    # Variables defined by the setup and potentially required throughout the analysis:
    #   HIGGS_CP_BASE
    #       The absolute analysis base directory. Used to infer file locations relative to it.
    #   HIGGS_CP_SETUP
    #       A flag that is set to 1 after the setup was successful.

    # prevent repeated setups
    if [ "${HIGGS_CP_SETUP}" = "1" ]; then
        >&2 echo "the higgs_cp analysis was already succesfully setup"
        >&2 echo "re-running the setup requires a new shell"
        return "1"
    fi


    #
    # prepare local variables
    #

    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"
    local orig="${PWD}"
    local setup_name="${1:-default}"
    local setup_is_default="false"
    [ "${setup_name}" = "default" ] && setup_is_default="true"

    # zsh options
    if ${shell_is_zsh}; then
        emulate -L bash
        setopt globdots
    fi


    #
    # global variables
    # (HIGGS_CP = higgs_cp, CF = columnflow)
    #

    # start exporting variables
    export HIGGS_CP_BASE="${this_dir}"
    export CF_BASE="${this_dir}/modules/columnflow"
    export CF_REPO_BASE="${HIGGS_CP_BASE}"
    export CF_REPO_BASE_ALIAS="HIGGS_CP_BASE"
    export CF_SETUP_NAME="${setup_name}"

    # load cf setup helpers
    CF_SKIP_SETUP="1" source "${CF_BASE}/setup.sh" "" || return "$?"

    # interactive setup
    if [ "${CF_REMOTE_ENV}" != "1" ]; then
        cf_setup_interactive_body() {
            # pre-export the CF_FLAVOR which will be cms
            export CF_FLAVOR="cms"

            # query common variables
            cf_setup_interactive_common_variables

            # query specific variables
            # nothing yet ...
        }
        cf_setup_interactive "${CF_SETUP_NAME}" "${HIGGS_CP_BASE}/.setups/${CF_SETUP_NAME}.sh" || return "$?"
    fi

    # continue the fixed setup
    export CF_CONDA_BASE="${CF_CONDA_BASE:-${CF_SOFTWARE_BASE}/conda}"
    export CF_VENV_BASE="${CF_VENV_BASE:-${CF_SOFTWARE_BASE}/venvs}"
    export CF_CMSSW_BASE="${CF_CMSSW_BASE:-${CF_SOFTWARE_BASE}/cmssw}"


    #
    # common variables
    #

    cf_setup_common_variables || return "$?"


    #
    # minimal local software setup
    #

    cf_setup_software_stack "${CF_SETUP_NAME}" || return "$?"

    # ammend paths that are not covered by the central cf setup
    export PATH="${HIGGS_CP_BASE}/bin:${PATH}"
    export PYTHONPATH="${HIGGS_CP_BASE}:${HIGGS_CP_BASE}/modules/cmsdb:${PYTHONPATH}"

    # initialze submodules
    if [ -e "${HIGGS_CP_BASE}/.git" ]; then
        local m
        for m in $( ls -1q "${HIGGS_CP_BASE}/modules" ); do
            cf_init_submodule "${HIGGS_CP_BASE}" "modules/${m}"
        done
    fi


    #
    # git hooks
    #

    cf_setup_git_hooks || return "$?"


    #
    # law setup
    #

    export LAW_HOME="${LAW_HOME:-${HIGGS_CP_BASE}/.law}"
    export LAW_CONFIG_FILE="${LAW_CONFIG_FILE:-${HIGGS_CP_BASE}/law.cfg}"

    if which law &> /dev/null; then
        # source law's bash completion scipt
        source "$( law completion )" ""

        # silently index
        law index -q
    fi

    # finalize
    export HIGGS_CP_SETUP="1"
}

main() {
    # Invokes the main action of this script, catches possible error codes and prints a message.

    # run the actual setup
    if setup_higgs_cp "$@"; then
        cf_color green "higgs_cp analysis successfully setup"
        return "0"
    else
        local code="$?"
        cf_color red "setup failed with code ${code}"
        return "${code}"
    fi
}

# entry point
if [ "${HIGGS_CP_SKIP_SETUP}" != "1" ]; then
    main "$@"
fi