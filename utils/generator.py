from keras.preprocessing.image import ImageDataGenerator
import configparser
config = configparser.RawConfigParser()
config.read('config.txt')
seed = int(config.get('model settings','seed'))
batch_size = int(config.get('train settings','batch_size'))
imgen = ImageDataGenerator(zoom_range=[0.5,1.5],
                           shear_range=2,
                           rotation_range=90,
                           width_shift_range=200,
                           height_shift_range=200,
                           horizontal_flip=True,
                           vertical_flip=True,
                           validation_split=0.2,
                           fill_mode='wrap')
def m_gen(raw,mask):
    rgen = imgen.flow(raw, batch_size=batch_size, shuffle=True, seed=seed)
    mgen = imgen.flow(mask, batch_size=batch_size, shuffle=True, seed=seed)
    while 1:
        r = rgen.next()
        m = mgen.next()
        yield r,m
def sv_gen(raw,soma,vessel):
    rgen = imgen.flow(raw, batch_size=batch_size, shuffle=True, seed=seed)
    sgen = imgen.flow(soma, batch_size=batch_size, shuffle=True, seed=seed)
    vgen = imgen.flow(vessel, batch_size=batch_size, shuffle=True, seed=seed)
    while 1:
        r = rgen.next()
        s = sgen.next()
        v = vgen.next()
        yield r,s,v