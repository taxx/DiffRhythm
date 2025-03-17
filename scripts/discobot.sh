cd "$(dirname "$0")"
cd ../

export PYTHONPATH=$PYTHONPATH:$PWD

export CUDA_VISIBLE_DEVICES=0
#--ref-prompt "classical genres, hopeful mood, piano." \
#    --lrc-path infer/example/eg_en.lrc \
#--ref-prompt "Arctic research station, theremin auroras dancing with geomagnetic storms"  \
python3 infer/discobot.py
