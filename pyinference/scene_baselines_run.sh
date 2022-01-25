for i in {0..399}
do 
    sbatch scene_baseline.sh $i -o results/baseline_retinanet/$i.out
done 
