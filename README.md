
vit_v1
- using nn.TransformerEncoderLayer


vit_v2
- no longer couple embedding dimension to patch size 
- implemented intuitive version of the encoder block 





Notes: 

For the first task (encoder only classification of MNIST using ViT):
https://youtu.be/Vonyoz6Yt9c?si=i7epLD2SwzvwV39K

But it uses nn.TransformerEncoderLayer as an abstraction 


For building the encoder: 


Good for intuition on attention: https://youtu.be/eMlx5fFNoYc?si=Wxte-nsXKWLjncni


Used this for the more intuitive implementation of multi-head self attention: https://medium.com/data-science/implementing-vision-transformer-vit-from-scratch-3e192c6155f0o


This implementation above is not what is used in practice. Use this instead: [add later]




Goal for Wed: 

1 pomo: 1.5h
-handle the dataset (30 min) 
    - ok implementation was done but I didn't get into all the details (I will skip for now and revisit)
- build out the decoder (1h)








Plan today: 
- move quite quickly, rely on chat to improve things



Keep track of this:
"You do not have to modify the encoder code; you will only have to re-create the model so the PatchEmbedding constructor knows the new num_patches. We will do that after we finish the decoder."








Goal for Thurs: 
- Clarify understanding (do the more optimised implementation)