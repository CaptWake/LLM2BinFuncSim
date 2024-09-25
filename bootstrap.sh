SUBMODULE_PATH="HermesSim"
INPUTS_URL="https://zenodo.org/records/10369788/files/inputs.tar.xz?download=1"
DBS_URL="https://zenodo.org/records/10369788/files/dbs.tar.xz?download=1"
INPUTS_FILENAME="inputs.tar.xz"
DBS_FILENAME="dbs.tar.xz"

# Function to log messages
log() {
    echo "[*] $1"
}

# Function to log errors
log_error() {
    echo "[!] $1"
}

# Function to handle errors
handle_error() {
    log_error "$1"
    # exit 1
}

# Initialize submodules
initialize_submodules() {
    log "Initializing submodules..."
    git submodule update --init --recursive || handle_error "Error cloning repository or initializing submodules."
}

# Apply patch to submodule
apply_patch() {
    log "Applying patch to the submodule..."
    (cd "$SUBMODULE_PATH" && git apply --stat --apply ../patch/hermessim.patch) || handle_error "Error applying patches to hermessim submodule."
}

# Move scripts
move_scripts() {
    log "Moving scripts..."
    mv patch/run.sh "$SUBMODULE_PATH" && mv patch/*.{sh,py} "${SUBMODULE_PATH}/preprocess"
}

# Download and extract data
download_and_extract() {
    log "Downloading data and extracting it..."
    mkdir -p "${SUBMODULE_PATH}/outputs/"
    mkdir -p "${SUBMODULE_PATH}/inputs/acfg_llm/"
    mkdir -p "${SUBMODULE_PATH}/dbs/"

    curl --output "$INPUTS_FILENAME" "$INPUTS_URL" || handle_error "Error downloading inputs file."
    tar -xvf "$INPUTS_FILENAME" -C "$SUBMODULE_PATH/inputs/" || handle_error "Error extracting inputs file."
    rm "$INPUTS_FILENAME"


    curl --output "$DBS_FILENAME" "$DBS_URL" || handle_error "Error downloading dbs file."
    tar -xvf "$DBS_FILENAME" -C "$SUBMODULE_PATH/dbs/" || handle_error "Error extracting dbs file."
    rm "$DBS_FILENAME"
}


initialize_submodules
apply_patch
move_scripts
download_and_extract
log "Setup completed successfully."