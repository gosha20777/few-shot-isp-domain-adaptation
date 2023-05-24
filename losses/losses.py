import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import Model


base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)


def saturation(img):
	mean = tf.keras.backend.mean(img, axis=-1, keepdims = True)
	mul = tf.constant([1,1,1,3], tf.int32)
	mean = tf.tile(mean, mul)
	img = tf.math.subtract(img, mean)
	sat = tf.einsum('aijk,aijk->aij', img, img)
	sat = tf.math.scalar_mul((1.0/3.0),sat)
	sat = tf.math.add(sat, tf.math.scalar_mul(1e-7, tf.ones_like(sat)))
	sat = tf.math.sqrt(sat)
	return sat

def get_exp(img,c):
	#cimg = tf.slice(img,[0,0,0,c],[img.get_shape()[0],img.get_shape()[1],img.get_shape()[2],1])
	cimg = tf.squeeze(img,axis=-1)
	m = tf.math.scalar_mul(0.5, tf.ones_like(cimg))
	cimg = tf.math.subtract(cimg,m)
	cimg = tf.math.multiply(cimg,cimg)
	cimg = tf.math.scalar_mul(-12.5,cimg)
	return cimg

def exposure(img):
	rimg, gimg, bimg = tf.split(img, num_or_size_splits=3, axis=-1)
	rimg = get_exp(rimg,0)
	gimg = get_exp(gimg,1)
	bimg = get_exp(bimg,2)
	img = tf.math.add(rimg,gimg)
	img = tf.math.add(img,bimg)
	exp = tf.math.exp(img)
	return exp

def contrast(img):
	mean = tf.keras.backend.mean(img, axis=-1, keepdims=True)
	lap_fil = [[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]]
	lap_fil = tf.expand_dims(lap_fil,-1)
	lap_fil = tf.expand_dims(lap_fil,-1)
	con = tf.nn.convolution(mean, lap_fil, padding='SAME')
	con = tf.math.abs(con)
	con = tf.squeeze(con,axis=-1)
	return con

def exp_map(img,pc,ps,pe):
	con = contrast(img)
	sat = saturation(img)
	exp = exposure(img)
	if pc!=1 or pe!=1 or ps!=1:
		pc = tf.math.scalar_mul(pc, tf.ones_like(con))
		ps = tf.math.scalar_mul(ps, tf.ones_like(con))
		pe = tf.math.scalar_mul(pe, tf.ones_like(con))
		con = tf.math.pow(con,pc)
		sat = tf.math.pow(sat,pe)
		exp = tf.math.pow(exp,ps)
	wt_map = tf.math.multiply(con,sat)
	wt_map = tf.math.multiply(wt_map,exp)
	return wt_map


def mssim(y_true, y_pred):
    costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return costs


def color(y_true, y_pred):
    ytn = tf.math.l2_normalize(y_true, axis = -1, epsilon=1e-9)
    ypn = tf.math.l2_normalize(y_pred, axis = -1, epsilon=1e-9)
    color_cos = tf.einsum('aijk,aijk->aij', ytn, ypn)
    ca_mean = 1.0 - tf.reduce_mean(color_cos)
    return ca_mean


def vgg_loss(y_true, y_pred):
    cost = tf.reduce_mean(tf.math.square(tf.math.subtract(vgg(y_true), vgg(y_pred))))
    return cost


def exp_fusion(y_true, y_pred):
    costs = tf.reduce_mean(tf.math.abs(tf.math.subtract(exp_map(y_true, 1, 1, 1), exp_map(y_pred, 1, 1, 1))))
    return costs