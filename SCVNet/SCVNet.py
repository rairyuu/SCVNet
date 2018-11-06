from __future__ import print_function
import sys
import re
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import png


torch.cuda.set_device(0)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

SF_image_width = 960
SF_image_height = 540

SF_image_width_train = 768
SF_image_height_train = 320

SF_crop_x_min = 0
SF_crop_x_max = SF_image_width - SF_image_width_train
SF_crop_y_min = 0
SF_crop_y_max = SF_image_height - SF_image_height_train

KITTI_image_width = 1224
KITTI_image_height = 370

KITTI_image_width_train = 768
KITTI_image_height_train = 320

KITTI_crop_x_min = 0
KITTI_crop_x_max = KITTI_image_width - KITTI_image_width_train
KITTI_crop_y_min = 0
KITTI_crop_y_max = KITTI_image_height - KITTI_image_height_train

batchsize = 1


def readPFM(file, ret_PIL=True):
	file = open(file, 'rb')

	color = None
	width = None
	height = None
	scale = None
	endian = None

	header = file.readline().rstrip().decode('utf-8')
	if header == 'PF':
		raise Exception('Only ONE channel image is supported.')
	elif header == 'Pf':
		color = False
	else:
		raise Exception('Not a PFM file.')

	dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
	if dim_match:
		width, height = map(int, dim_match.groups())
	else:
		raise Exception('Malformed PFM header.')

	scale = float(file.readline().rstrip())
	if scale < 0:  # little-endian
		endian = '<'
		scale = -scale
	else:
		endian = '>'  # big-endian

	npImage = np.fromfile(file, endian + 'f')
	shape = (height, width)

	npImage = np.reshape(npImage, shape)
	npImage = np.flipud(npImage)

	if ret_PIL:
		img = Image.fromarray(npImage, 'F')
		return img

	return npImage


def writePFM(file, img, scale=1):
	file = open(file, 'wb')

	color = None

	if isinstance(img, Image.Image):
		npImage = np.array(img)
	else:
		npImage = img.numpy()

	if len(npImage.shape) == 3:
		npImage = np.transpose(npImage, (1, 2, 0))

	if npImage.dtype.name != 'float32':
		raise Exception('Image dtype must be float32.')

	npImage = np.flipud(npImage)

	if len(npImage.shape) == 3 and npImage.shape[2] == 3:  # color image
		raise Exception('Image must have H x W x 1 or H x W dimensions.')
	elif len(npImage.shape) == 2 or (len(npImage.shape) == 3 and npImage.shape[2] == 1):  # greyscale
		color = False
	else:
		raise Exception('Image must have H x W x 1 or H x W dimensions.')

	file.write('Pf\n'.encode('utf-8'))
	file.write(('%d %d\n' % (npImage.shape[1], npImage.shape[0])).encode('utf-8'))

	endian = npImage.dtype.byteorder

	if endian == '<' or endian == '=' and sys.byteorder == 'little':
		scale = -scale

	file.write(('%f\n' % scale).encode('utf-8'))

	npImage.tofile(file)

	return


def loadPNG16(file):
	pngReader = png.Reader(filename=file)
	pngData = pngReader.read()[2]
	npImage = np.vstack(map(np.uint16, pngData))

	return npImage.astype(np.float32) / 256.0


def savePNG16(file, img):
	with open(file, 'wb') as fileOpened:
		img = img * 256.0
		npImage = img.numpy().astype(np.uint16)
		pngWriter = png.Writer(width=npImage.shape[1], height=npImage.shape[0], bitdepth=16, greyscale=True)
		pngWriter.write(fileOpened, npImage.tolist())

	return


def loadImage(file):
	return Image.open(file)


def saveImage(file, img):
	img = img * 0.5 + 0.5
	npImage = img.numpy()
	plt.imsave(file, np.transpose(npImage, (1, 2, 0)))
	return


class SFDatasetTrain(data.Dataset):
	def __init__(self, transform=None, transform_label=None):
		self.root = SCENE_FLOW_TRAIN_PATH_IMAGE
		self.root_label = SCENE_FLOW_TRAIN_PATH_LABEL
		self.group = [
			'A/',
			'B/',
			'C/'
		]
		self.camera = [
			'left/',
			'right/'
		]

		if transform is None:
			self.transform = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				]
			)
		else:
			self.transform = transform

		self.transform_label = transform_label

		return

	def __getitem__(self, item):
		f0 = random.randint(0, 2)
		f1 = random.randint(0, 749)
		f2 = random.randint(6, 15)

		file_left = self.root + self.group[f0] + '%04d/' % f1 + self.camera[0] + '%04d.png' % f2
		file_right = self.root + self.group[f0] + '%04d/' % f1 + self.camera[1] + '%04d.png' % f2
		file_label = self.root_label + self.group[f0] + '%04d/' % f1 + self.camera[0] + '%04d.pfm' % f2

		image_left = loadImage(file_left)
		image_right = loadImage(file_right)
		disparity_left = readPFM(file_label, ret_PIL=False)

		crop_x = random.randint(SF_crop_x_min, SF_crop_x_max)
		crop_y = random.randint(SF_crop_y_min, SF_crop_y_max)
		image_left_crop = image_left.crop(
			(
				crop_x,
				crop_y,
				crop_x + SF_image_width_train,
				crop_y + SF_image_height_train
			)
		)
		image_right_crop = image_right.crop(
			(
				crop_x,
				crop_y,
				crop_x + SF_image_width_train,
				crop_y + SF_image_height_train
			)
		)
		disparity_left_crop = disparity_left[crop_y:(crop_y + SF_image_height_train), crop_x:(crop_x + SF_image_width_train)]

		if self.transform is not None:
			image_left_crop = self.transform(image_left_crop)
			image_right_crop = self.transform(image_right_crop)
		if self.transform_label is not None:
			disparity_left_crop = self.transform_label(disparity_left_crop)
		else:
			disparity_left_crop = torch.from_numpy(disparity_left_crop.copy())

		return image_left_crop, image_right_crop, disparity_left_crop

	def __len__(self):
		return 210000


class SFDatasetTest(data.Dataset):
	def __init__(self, transform=None, transform_label=None):
		self.root = SCENE_FLOW_TEST_PATH_IMAGE
		self.root_label = SCENE_FLOW_TEST_PATH_LABEL
		self.group = [
			'A/',
			'B/',
			'C/'
		]
		self.camera = [
			'left/',
			'right/'
		]

		if transform is None:
			self.transform = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				]
			)
		else:
			self.transform = transform

		self.transform_label = transform_label

		return

	def __getitem__(self, item):
		f0 = random.randint(0, 2)
		f1 = random.randint(0, 149)
		f2 = random.randint(6, 15)

		file_left = self.root + self.group[f0] + '%04d/' % f1 + self.camera[0] + '%04d.png' % f2
		file_right = self.root + self.group[f0] + '%04d/' % f1 + self.camera[1] + '%04d.png' % f2
		file_label = self.root_label + self.group[f0] + '%04d/' % f1 + self.camera[0] + '%04d.pfm' % f2

		image_left = loadImage(file_left)
		image_right = loadImage(file_right)
		disparity_left = readPFM(file_label, ret_PIL=False)

		if self.transform is not None:
			image_left = self.transform(image_left)
			image_right = self.transform(image_right)

		if self.transform_label is not None:
			disparity_left = self.transform_label(disparity_left)
		else:
			disparity_left = torch.from_numpy(disparity_left.copy())

		return image_left, image_right, disparity_left

	def __len__(self):
		return 4500


class KITTIDatasetTrain(data.Dataset):
	def __init__(self, transform=None, transform_label=None):
		self.root = KITTI_2015_TRAIN_PATH_IMAGE
		self.root_label = KITTI_2015_TRAIN_PATH_LABEL
		self.camera = [
			'image_2/',
			'image_3/'
		]

		if transform is None:
			self.transform = transforms.Compose(
				[
					transforms.ToTensor()
				]
			)
		else:
			self.transform = transform

		self.transform_label = transform_label

		return

	def __getitem__(self, item):
		f0 = item % 200

		file_left = self.root + self.camera[0] + '%06d_10.png' % f0
		file_right = self.root + self.camera[1] + '%06d_10.png' % f0
		file_label = self.root_label + '%06d_10.png' % f0

		image_left = loadImage(file_left)
		image_right = loadImage(file_right)
		disparity_left = loadPNG16(file_label)

		crop_x = random.randint(KITTI_crop_x_min, KITTI_crop_x_max)
		crop_y = random.randint(KITTI_crop_y_min, KITTI_crop_y_max)
		image_left_crop = image_left.crop(
			(
				crop_x,
				crop_y,
				crop_x + KITTI_image_width_train,
				crop_y + KITTI_image_height_train
			)
		)
		image_right_crop = image_right.crop(
			(
				crop_x,
				crop_y,
				crop_x + KITTI_image_width_train,
				crop_y + KITTI_image_height_train
			)
		)
		disparity_left_crop = disparity_left[crop_y:(crop_y + KITTI_image_height_train), crop_x:(crop_x + KITTI_image_width_train)]

		if self.transform is not None:
			image_left_crop = self.transform(image_left_crop)
			image_right_crop = self.transform(image_right_crop)
		if self.transform_label is not None:
			disparity_left_crop = self.transform_label(disparity_left_crop)
		else:
			disparity_left_crop = torch.from_numpy(disparity_left_crop.copy())

		exposure_scale = random.randint(900, 1100) * 0.001
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_left_crop[0] = (image_left_crop[0] * color_scale - 0.5) / 0.5
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_left_crop[1] = (image_left_crop[1] * color_scale - 0.5) / 0.5
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_left_crop[2] = (image_left_crop[2] * color_scale - 0.5) / 0.5

		exposure_scale = random.randint(900, 1100) * 0.001
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_right_crop[0] = (image_right_crop[0] * color_scale - 0.5) / 0.5
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_right_crop[1] = (image_right_crop[1] * color_scale - 0.5) / 0.5
		color_scale = random.randint(900, 1100) * 0.001 * exposure_scale
		image_right_crop[2] = (image_right_crop[2] * color_scale - 0.5) / 0.5

		return image_left_crop, image_right_crop, disparity_left_crop

	def __len__(self):
		return 20000


class KITTIDatasetTest(data.Dataset):
	def __init__(self, transform=None, transform_label=None):
		self.root = KITTI_2015_TEST_PATH_IMAGE
		self.root_label = KITTI_2015_TEST_PATH_LABEL
		self.camera = [
			'image_2/',
			'image_3/'
		]

		if transform is None:
			self.transform = transforms.Compose(
				[
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
				]
			)
		else:
			self.transform = transform

		self.transform_label = transform_label

		return

	def __getitem__(self, item):
		f0 = item

		file_left = self.root + self.camera[0] + '%06d_10.png' % f0
		file_right = self.root + self.camera[1] + '%06d_10.png' % f0
		file_label = self.root_label + '%06d_10.png' % f0

		image_left = loadImage(file_left)
		image_right = loadImage(file_right)
		disparity_left = loadPNG16(file_label)

		crop_x = 0
		crop_y = disparity_left.shape[0] - 352
		image_left_crop = image_left.crop(
			(
				crop_x,
				crop_y,
				crop_x + 1216,
				crop_y + 352
			)
		)
		image_right_crop = image_right.crop(
			(
				crop_x,
				crop_y,
				crop_x + 1216,
				crop_y + 352
			)
		)
		disparity_left_crop = disparity_left[crop_y:(crop_y + 352), crop_x:(crop_x + 1216)]
		# 1216 % 32 = 0, 352 % 32 = 0
		# only for sample

		if self.transform is not None:
			image_left_crop = self.transform(image_left_crop)
			image_right_crop = self.transform(image_right_crop)
		if self.transform_label is not None:
			disparity_left_crop = self.transform_label(disparity_left_crop)
		else:
			disparity_left_crop = torch.from_numpy(disparity_left_crop.copy())

		return image_left_crop, image_right_crop, disparity_left_crop

	def __len__(self):
		return 200


class SCVNet_UF(nn.Module):
	def __init__(self):
		super(SCVNet_UF, self).__init__()

		self.conv1 = nn.utils.weight_norm(nn.Conv2d(3, 32, 5, stride=2, padding=2), dim=0)

		self.conv2 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv3 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv4 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv5 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv6 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv7 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv8 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv9 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv10 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv11 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv12 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv13 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv14 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv15 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv16 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)
		self.conv17 = nn.utils.weight_norm(nn.Conv2d(32, 32, 3, padding=1), dim=0)

		self.conv18 = nn.Conv2d(64, 32, 3, padding=1)

		return

	def forward(self, x):
		y_1 = F.relu(self.conv1(x))

		y_2 = F.relu(self.conv2(y_1))
		y_3 = F.relu(y_1 + self.conv3(y_2))

		y_4 = F.relu(self.conv4(y_3))
		y_5 = F.relu(y_3 + self.conv5(y_4))

		y_6 = F.relu(self.conv6(y_5))
		y_7 = F.relu(y_5 + self.conv7(y_6))

		y_8 = F.relu(self.conv8(y_7))
		y_9 = F.relu(y_7 + self.conv9(y_8))

		y_10 = F.relu(self.conv10(y_9))
		y_11 = F.relu(y_9 + self.conv11(y_10))

		y_12 = F.relu(self.conv12(y_11))
		y_13 = F.relu(y_11 + self.conv13(y_12))

		y_14 = F.relu(self.conv14(y_13))
		y_15 = F.relu(y_13 + self.conv15(y_14))

		y_16 = F.relu(self.conv16(y_15))
		y_17 = F.relu(y_15 + self.conv17(y_16))

		x_18 = [y_7, y_17]
		x_18 = torch.cat(x_18, dim=1)
		y_18 = self.conv18(x_18)

		return y_18


class SCVNet(nn.Module):
	def __init__(self):
		super(SCVNet, self).__init__()

		self.UF18 = SCVNet_UF()
		self.pad_left = 127
		self.pad18 = nn.ZeroPad2d((self.pad_left, 0, 0, 0))

		self.conv19 = nn.utils.weight_norm(nn.Conv2d(64, 32, (3, 5), padding=(1, 2)), dim=0)
		self.conv19M = nn.utils.weight_norm(nn.Conv2d(32, 32, (3, 5), padding=(1, 2)), dim=0)
		self.conv20 = nn.utils.weight_norm(nn.Conv2d(32, 32, (3, 5), padding=(1, 2)), dim=0)

		self.conv21 = nn.utils.weight_norm(nn.Conv2d(64, 64, 5, stride=2, padding=2), dim=0)
		self.conv22 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv22M = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv23 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)

		self.conv24 = nn.utils.weight_norm(nn.Conv2d(64, 64, 5, stride=2, padding=2), dim=0)
		self.conv25 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv25M = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv26 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)

		self.conv27 = nn.utils.weight_norm(nn.Conv2d(64, 64, 5, stride=2, padding=2), dim=0)
		self.conv28 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv28M = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)
		self.conv29 = nn.utils.weight_norm(nn.Conv2d(64, 64, (3, 5), padding=(1, 2)), dim=0)

		self.conv30 = nn.utils.weight_norm(nn.Conv2d(64, 128, 5, stride=2, padding=2), dim=0)
		self.conv31 = nn.utils.weight_norm(nn.Conv2d(128, 128, (3, 5), padding=(1, 2)), dim=0)
		self.conv31M = nn.utils.weight_norm(nn.Conv2d(128, 128, (3, 5), padding=(1, 2)), dim=0)
		self.conv32 = nn.utils.weight_norm(nn.Conv2d(128, 128, (3, 5), padding=(1, 2)), dim=0)

		self.tConv33 = nn.utils.weight_norm(nn.ConvTranspose2d(128, 64, (3, 5), stride=2, padding=(1, 2), output_padding=1), dim=1)

		self.tConv34 = nn.utils.weight_norm(nn.ConvTranspose2d(64, 64, (3, 5), stride=2, padding=(1, 2), output_padding=1), dim=1)

		self.tConv35 = nn.utils.weight_norm(nn.ConvTranspose2d(64, 64, (3, 5), stride=2, padding=(1, 2), output_padding=1), dim=1)

		self.tConv36 = nn.utils.weight_norm(nn.ConvTranspose2d(64, 32, (3, 5), stride=2, padding=(1, 2), output_padding=1), dim=1)

		self.tConv37 = nn.ConvTranspose2d(32, 6, (3, 5), stride=2, padding=(1, 2), output_padding=1)

		return

	def forward(self, x_l, x_r):
		x = [x_l, x_r]
		x = torch.cat(x, dim=0)
		del x_l
		del x_r
		x = self.UF18.forward(x)
		image_width_half = x.size()[3]
		x_l = x[0].unsqueeze(0)
		x_r = self.pad18(x[1].unsqueeze(0))
		del x

		y_19 = []
		y_21 = []

		for i in range(32):
			crop_left = self.pad_left - i * 3
			x = [x_l, x_r[:, :, :, crop_left:(crop_left + image_width_half)]]
			x = torch.cat(x, dim=1)
			y_19.append(self.conv19(x))
			y_21.append(self.conv21(x))

		del x_l
		del x_r
		y_19 = F.relu(torch.cat(y_19, dim=0))
		y_19 = F.relu(self.conv19M(y_19))
		y_20 = F.relu(self.conv20(y_19))
		del y_19

		y_21 = F.relu(torch.cat(y_21, dim=0))
		y_22 = F.relu(self.conv22(y_21))
		y_22 = F.relu(self.conv22M(y_22))
		y_23 = F.relu(self.conv23(y_22))
		del y_22

		y_24 = F.relu(self.conv24(y_21))
		del y_21
		y_25 = F.relu(self.conv25(y_24))
		y_25 = F.relu(self.conv25M(y_25))
		y_26 = F.relu(self.conv26(y_25))
		del y_25

		y_27 = F.relu(self.conv27(y_24))
		del y_24
		y_28 = F.relu(self.conv28(y_27))
		y_28 = F.relu(self.conv28M(y_28))
		y_29 = F.relu(self.conv29(y_28))
		del y_28

		y_30 = F.relu(self.conv30(y_27))
		del y_27
		y_31 = F.relu(self.conv31(y_30))
		del y_30
		y_31 = F.relu(self.conv31M(y_31))
		y_32 = F.relu(self.conv32(y_31))
		del y_31

		y_33 = F.relu(y_29 + self.tConv33(y_32))
		del y_29
		del y_32
		y_34 = F.relu(y_26 + self.tConv34(y_33))
		del y_26
		del y_33
		y_35 = F.relu(y_23 + self.tConv35(y_34))
		del y_23
		del y_34
		y_36 = F.relu(y_20 + self.tConv36(y_35))
		del y_20
		del y_35
		y_37 = self.tConv37(y_36)
		del y_36

		final_result = [y_37[0, 3:, :, :]]
		for i in range(1, 32):
			final_result.append(y_37[i, :, :, :])
		final_result = torch.cat(final_result, dim=0)
		del y_37

		final_result = final_result.unsqueeze(0)
		final_result = final_result.unsqueeze(0)

		return final_result


trainSetSF = SFDatasetTrain()
trainLoaderSF = data.DataLoader(
	trainSetSF,
	batch_size=batchsize,
	shuffle=False,
	num_workers=1,
	drop_last=True
)

testSetSF = SFDatasetTest()
testLoaderSF = data.DataLoader(
	testSetSF,
	batch_size=1,
	shuffle=False,
	num_workers=1,
	drop_last=True
)

trainSetKITTI = KITTIDatasetTrain()
trainLoaderKITTI = data.DataLoader(
	trainSetKITTI,
	batch_size=batchsize,
	shuffle=False,
	num_workers=1,
	drop_last=True
)

testSetKITTI = KITTIDatasetTest()
testLoaderKITTI = data.DataLoader(
	testSetKITTI,
	batch_size=1,
	shuffle=False,
	num_workers=1,
	drop_last=True
)

net = SCVNet().cuda()
net.train()

disparity_array_fast_full = torch.FloatTensor([i for i in range(189)])
disparity_array_fast_full = Variable(disparity_array_fast_full.cuda(), requires_grad=False)


# Scene Flow
def NetTestSF():
	net.eval()

	resultSum1 = 0.0
	resultSum3 = 0.0
	resultSum5 = 0.0
	resultSum = 0.0

	TempPad32 = nn.ZeroPad2d((0, 0, 0, 4))

	for i, dataTest in enumerate(testLoaderSF, 0):
		img_ls, img_rs, d_ls = dataTest
		img_ls = Variable(img_ls.cuda(), volatile=True)
		img_rs = Variable(img_rs.cuda(), volatile=True)
		d_ls = Variable(d_ls.cuda(), volatile=True)
		img_ls = TempPad32(img_ls)
		img_rs = TempPad32(img_rs)
		result = net.forward(img_ls, img_rs)
		del img_ls
		del img_rs
		result = F.softmax(result[:, 0, :, :540, :], dim=1)
		result = torch.transpose(result, 1, 2)
		result = torch.transpose(result, 2, 3)
		result = torch.matmul(result, disparity_array_fast_full)
		result = torch.abs(result - d_ls)

		resultSum += torch.mean(result)

		result1 = torch.ge(result, 1.0)
		result1 = result1.type(torch.cuda.FloatTensor)
		result1 = torch.mean(result1)
		result1.cpu()
		resultSum1 += result1.data[0]

		result3 = torch.ge(result, 3.0)
		result3 = result3.type(torch.cuda.FloatTensor)
		result3 = torch.mean(result3)
		result3.cpu()
		resultSum3 += result3.data[0]

		result5 = torch.ge(result, 5.0)
		result5 = result5.type(torch.cuda.FloatTensor)
		result5 = torch.mean(result5)
		result5.cpu()
		resultSum5 += result5.data[0]

	resultSum1 /= 4500
	resultSum1 *= 100.0
	resultSum3 /= 4500
	resultSum3 *= 100.0
	resultSum5 /= 4500
	resultSum5 *= 100.0
	resultSum /= 4500

	print('1PE: %f%%, 3PE: %f%%, 5PE: %f%%, EPE: %f' % (resultSum1, resultSum3, resultSum5, resultSum))

	net.train()

	return


optimizer = optim.RMSprop(net.parameters(), lr=0.0001)

runningLoss = 0.0
for i, dataTrain in enumerate(trainLoaderSF, 0):
	img_ls, img_rs, d_ls = dataTrain
	img_ls = Variable(img_ls.cuda(), requires_grad=False)
	img_rs = Variable(img_rs.cuda(), requires_grad=False)
	d_ls = Variable(d_ls.cuda(), requires_grad=False)
	optimizer.zero_grad()
	result = net.forward(img_ls, img_rs)
	del img_ls
	del img_rs
	result = F.softmax(result[:, 0, :, :, :], dim=1)
	result = torch.transpose(result, 1, 2)
	result = torch.transpose(result, 2, 3)
	result = torch.matmul(result, disparity_array_fast_full)
	result = torch.abs(result - d_ls)
	d_ls1 = torch.ge(d_ls, 0.001)
	d_ls1 = d_ls1.type(torch.cuda.FloatTensor)
	d_ls2 = torch.le(d_ls, 188.0)
	d_ls2 = d_ls2.type(torch.cuda.FloatTensor)
	d_ls = torch.mul(d_ls1, d_ls2)
	result = torch.mul(result, d_ls)
	result = torch.sum(result) / torch.sum(d_ls)
	del d_ls
	del d_ls1
	del d_ls2
	result.backward()
	optimizer.step()
	runningLoss += result.data.cpu()[0]
	del result
	if i % 100 == 99:
		if i % 10000 == 9999:
			torch.save(net.state_dict(), 'params_SCV_Net_%d.pkl' % (i + 1))
		print('[SCV-Net-0, %d] loss: %.3f' % (i + 1, runningLoss / 100))
		runningLoss = 0.0

NetTestSF()

print('Training SCV-Net-0 finished.')

# KITTI 2015
optimizer = optim.RMSprop(net.parameters(), lr=0.0002)

runningLoss = 0.0
for i, dataTrain in enumerate(trainLoaderKITTI, 0):
	img_ls, img_rs, d_ls = dataTrain
	img_ls = Variable(img_ls.cuda(), requires_grad=False)
	img_rs = Variable(img_rs.cuda(), requires_grad=False)
	d_ls = Variable(d_ls.cuda(), requires_grad=False)
	optimizer.zero_grad()
	result = net.forward(img_ls, img_rs)
	del img_ls
	del img_rs
	result = F.softmax(result[:, 0, :, :, :], dim=1)
	result = torch.transpose(result, 1, 2)
	result = torch.transpose(result, 2, 3)
	result = torch.matmul(result, disparity_array_fast_full)
	result = torch.abs(result - d_ls)
	d_ls1 = torch.ge(d_ls, 0.001)
	d_ls1 = d_ls1.type(torch.cuda.FloatTensor)
	d_ls2 = torch.le(d_ls, 188.0)
	d_ls2 = d_ls2.type(torch.cuda.FloatTensor)
	d_ls = torch.mul(d_ls1, d_ls2)
	result = torch.mul(result, d_ls)
	result = torch.sum(result) / torch.sum(d_ls)
	del d_ls
	del d_ls1
	del d_ls2
	result.backward()
	optimizer.step()
	runningLoss += result.data.cpu()[0]
	del result
	if i % 100 == 99:
		if i % 1000 == 999:
			torch.save(net.state_dict(), 'params_FT_SCV_Net_%d.pkl' % (i + 1 + 210000))
		print('[FT-SCV-Net-1, %d] loss: %.3f' % (i + 1 + 210000, runningLoss / 100))
		runningLoss = 0.0

print('Training FT-SCV-Net-1 finished.')

optimizer = optim.RMSprop(net.parameters(), lr=0.0001)

runningLoss = 0.0
for i, dataTrain in enumerate(trainLoaderKITTI, 0):
	img_ls, img_rs, d_ls = dataTrain
	img_ls = Variable(img_ls.cuda(), requires_grad=False)
	img_rs = Variable(img_rs.cuda(), requires_grad=False)
	d_ls = Variable(d_ls.cuda(), requires_grad=False)
	optimizer.zero_grad()
	result = net.forward(img_ls, img_rs)
	del img_ls
	del img_rs
	result = F.softmax(result[:, 0, :, :, :], dim=1)
	result = torch.transpose(result, 1, 2)
	result = torch.transpose(result, 2, 3)
	result = torch.matmul(result, disparity_array_fast_full)
	result = torch.abs(result - d_ls)
	d_ls1 = torch.ge(d_ls, 0.001)
	d_ls1 = d_ls1.type(torch.cuda.FloatTensor)
	d_ls2 = torch.le(d_ls, 188.0)
	d_ls2 = d_ls2.type(torch.cuda.FloatTensor)
	d_ls = torch.mul(d_ls1, d_ls2)
	result = torch.mul(result, d_ls)
	result = torch.sum(result) / torch.sum(d_ls)
	del d_ls
	del d_ls1
	del d_ls2
	result.backward()
	optimizer.step()
	runningLoss += result.data.cpu()[0]
	del result
	if i % 100 == 99:
		if i % 1000 == 999:
			torch.save(net.state_dict(), 'params_FT_SCV_Net_%d.pkl' % (i + 1 + 230000))
		print('[FT-SCV-Net-2, %d] loss: %.3f' % (i + 1 + 230000, runningLoss / 100))
		runningLoss = 0.0

print('Training FT-SCV-Net-2 finished.')

optimizer = optim.RMSprop(net.parameters(), lr=0.00005)

runningLoss = 0.0
for i, dataTrain in enumerate(trainLoaderKITTI, 0):
	img_ls, img_rs, d_ls = dataTrain
	img_ls = Variable(img_ls.cuda(), requires_grad=False)
	img_rs = Variable(img_rs.cuda(), requires_grad=False)
	d_ls = Variable(d_ls.cuda(), requires_grad=False)
	optimizer.zero_grad()
	result = net.forward(img_ls, img_rs)
	del img_ls
	del img_rs
	result = F.softmax(result[:, 0, :, :, :], dim=1)
	result = torch.transpose(result, 1, 2)
	result = torch.transpose(result, 2, 3)
	result = torch.matmul(result, disparity_array_fast_full)
	result = torch.abs(result - d_ls)
	d_ls1 = torch.ge(d_ls, 0.001)
	d_ls1 = d_ls1.type(torch.cuda.FloatTensor)
	d_ls2 = torch.le(d_ls, 188.0)
	d_ls2 = d_ls2.type(torch.cuda.FloatTensor)
	d_ls = torch.mul(d_ls1, d_ls2)
	result = torch.mul(result, d_ls)
	result = torch.sum(result) / torch.sum(d_ls)
	del d_ls
	del d_ls1
	del d_ls2
	result.backward()
	optimizer.step()
	runningLoss += result.data.cpu()[0]
	del result
	if i % 100 == 99:
		if i % 1000 == 999:
			torch.save(net.state_dict(), 'params_FT_SCV_Net_%d.pkl' % (i + 1 + 250000))
		print('[FT-SCV-Net-3, %d] loss: %.3f' % (i + 1 + 250000, runningLoss / 100))
		runningLoss = 0.0

print('Training FT-SCV-Net-3 finished.')

print('Training finished.')
