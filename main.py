def get_LPIPS_distance(true_img, fake_img):
    import tensorflow.compat.v1 as tf  # 用tf1.x版本的功能
    tf.disable_v2_behavior()  # 禁用2.x版本的功能
    from utils.lpips import lpips_tf
    # true_img & fake_img為np array型態 shape=(batch,w,h,channel)
    image0_ph = tf.convert_to_tensor(true_img, dtype=tf.float32, name="my_tensor")
    image1_ph = tf.convert_to_tensor(fake_img, dtype=tf.float32, name="my_tensor")
    distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')

    with tf.Session() as session:  # tf.Session是tf 1.x版本限定，2.x版本不能使用
        distance = session.run(distance_t, feed_dict={image0_ph: true_img, image1_ph: fake_img})
    return distance
if __name__=='__main__':
		# img1與img2請依照實際情況使用
		print(get_LPIPS_distance(img1, img2))
