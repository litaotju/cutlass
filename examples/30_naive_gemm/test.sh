for m in `seq 2048 16 2064`; do
    for n in `seq 2048 16 2064`; do
        for k in `seq 256 256 512`; do
            echo ./tensor_gemm -m $m -n $n -k $k
            ./tensor_gemm -m $m -n $n -k $k
        done
    done
done
