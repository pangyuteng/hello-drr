FROM huggingface/transformers-pytorch-gpu

RUN pip install diffdrr pandas pydicom pylibjpeg[libjpeg,openjpeg,rle]