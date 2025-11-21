tar xf CUB_200_2011.tgz -C data
rm CUB_200_2011.tgz
cd data
python split_cub_dataset.py
rm -r CUB_200_2011
rm attributes.txt
cd ..