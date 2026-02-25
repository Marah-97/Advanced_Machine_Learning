for i in 1 2 3 4 5 6 7 8 9 10
do
    echo "Starting run $i"
    python src/vae_gaussian_prior.py --run $i --device cuda
done