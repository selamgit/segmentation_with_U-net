from model import *
from data import *

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
# myGene = trainGenerator(2,'data/material/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(2,'data/material/train','image','label',data_gen_args,save_to_dir = 'data/material/train/aug')

model = unet()
model_checkpoint = ModelCheckpoint('unet_material.hdf5', monitor='loss',verbose=1, save_best_only=False)
model.fit_generator(myGene,steps_per_epoch=300,epochs=20,callbacks=[model_checkpoint])
# # # #
testGene = testGenerator("data/material/test")
results = model.predict_generator(testGene, 400, verbose=1)
saveResult("data/material/test", results)
