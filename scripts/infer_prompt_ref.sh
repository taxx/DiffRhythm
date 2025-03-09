cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES=0
#--ref-prompt "classical genres, hopeful mood, piano." \
#    --lrc-path infer/example/eg_en.lrc \
#--ref-prompt "Arctic research station, theremin auroras dancing with geomagnetic storms"  \
#--ref-prompt "Futuristic cityscape neon reflections, hovercars zipping through the night." \
python3 infer/infer.py \
    --ref-prompt "Futuristic cityscape neon reflections, hovercars zipping through the night." \
    --audio-length 95 \
    --lrc-path infer/example/eg_en.lrc \
    --repo_id ASLP-lab/DiffRhythm-base \
    --output-dir infer/example/output \
    --chunked
