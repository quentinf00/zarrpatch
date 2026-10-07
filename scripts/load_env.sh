# Sourced by pixi on activation: export the variables of the gitignored .env.
if [ -f "${PIXI_PROJECT_ROOT:-.}/.env" ]; then
    set -a
    . "${PIXI_PROJECT_ROOT:-.}/.env"
    set +a
fi
