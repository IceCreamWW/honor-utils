set -e


num_noises="1,1"
snr="0,10"
# wav_scp="/mnt/lustre/sjtu/home/syz24/myAna/ww089/wenet/examples/wenetspeech/s0/data/train_m/wav.scp"
# segments="/mnt/lustre/sjtu/home/syz24/myAna/ww089/wenet/examples/wenetspeech/s0/data/train_m/segments"
# output="/mnt/lustre/sjtu/home/syz24/myAna/ww089/data/wenet/train_m_musan_music_noise_snr_0_10db_1_noise"
wav_scp=
segments=
noise_scps="/mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/kaldi/music.scp /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/kaldi/noise.scp" 
output=
nj=16

. ./path.sh
. utils/parse_options.sh


echo "splitting wav.scp"
split_scps=""
for i in `seq 1 $nj`; do
    mkdir -p $output/$i
    split_scps="$split_scps $output/$i/spk1.scp"
done
utils/split_scp.pl $wav_scp $split_scps

if ! [ -z $segments ]; then
    echo "splitting segments"
    for i in `seq 1 $nj`; do
        awk 'NR==FNR{a[$1]=1} NR!=FNR{if($2 in a) print $0}' $output/$i/spk1.scp $segments > $output/$i/segments
    done
    cp $segments $output/segments
fi

echo "adding noise"

if [ -z $segments ]; then
    utils/run.pl JOB=1:$nj $output/JOB/log/add_noise.JOB.log \
        python local/utils/add_noise.py \
            --snr=$snr \
            --wav-scp $output/JOB/spk1.scp \
            --noise-scps $noise_scps \
            --output $output/JOB
else
    utils/run.pl JOB=1:$nj $output/JOB/log/add_noise.JOB.log \
        python local/utils/add_noise.py \
            --snr=$snr \
            --wav-scp $output/JOB/spk1.scp \
            --segments $output/JOB/segments \
            --noise-scps $noise_scps \
            --output $output/JOB
fi

echo "combining wav.scp"
for i in `seq 1 $nj`; do
    cat $output/$i/wav.scp
done > $output/wav.scp

if [ -z $segments ]; then
    echo "copy spk1.scp"
    cp $wav_scp $output/spk1.scp
else
    echo "combining spk1.scp"
    for i in `seq 1 $nj`; do
        cat $output/$i/spk1.scp
    done > $output/spk1.scp
fi


echo "Done"


