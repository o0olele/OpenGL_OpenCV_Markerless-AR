#include "camera.h"

#define MarkerLength 1.75

void Camera::setProjection()
{
	float farp = 100, nearp = 0.1;
	projection_matrix_= Mat::zeros(4, 4, CV_32F);

	float f_x = camera_matrix_.at<double>(0, 0);
	float f_y = camera_matrix_.at<double>(1, 1);

	float c_x = camera_matrix_.at<double>(0, 2);
	float c_y = camera_matrix_.at<double>(1, 2);

	projection_matrix_.at<float>(0, 0) = 2 * f_x / (float)fwidth_;
	projection_matrix_.at<float>(1, 1) = 2 * f_y / (float)fheight_;

	projection_matrix_.at<float>(2, 0) = 1.0f - 2 * c_x / (float)fwidth_;
	projection_matrix_.at<float>(2, 1) = 2 * c_y / (float)fheight_ - 1.0f;
	projection_matrix_.at<float>(2, 2) = -(farp + nearp) / (farp - nearp);
	projection_matrix_.at<float>(2, 3) = -1.0f;

	projection_matrix_.at<float>(3, 2) = -2.0f*farp*nearp / (farp - nearp);
}

Camera::Camera(shared_ptr<Config> config_ptr)
{
	config_ptr->get("image_width", fwidth_);
	config_ptr->get("image_height", fheight_);
	config_ptr->get("camera_matrix", camera_matrix_);
	config_ptr->get("distortion_coefficients", dist_coeffs_);

	dictionary_ = getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(0));

	setProjection();
}

Camera::~Camera()
{
}

bool Camera::marker_based_compute(Mat &image)
{
	Ptr<DetectorParameters> detectorParams = DetectorParameters::create();

	vector< int > markerIds;
	vector< vector<cv::Point2f> > markerCorners, rejectedCandidates;

	cv::aruco::detectMarkers(image, dictionary_, markerCorners, markerIds, detectorParams, rejectedCandidates);

	if (markerIds.size() > 0) {
		std::vector< cv::Vec3d > rvecs, tvecs;

		cv::aruco::drawDetectedMarkers(image, markerCorners, markerIds);
		cv::aruco::estimatePoseSingleMarkers(markerCorners, MarkerLength, camera_matrix_, dist_coeffs_, rvecs, tvecs);

		for (unsigned int i = 0; i < markerIds.size(); i++) {
			cv::Mat rot;
			cv::Vec3d r = rvecs[i];
			cv::Vec3d t = tvecs[i];

			view_matrix_ = cv::Mat::zeros(4, 4, CV_32F);

			Rodrigues(rvecs[i], rot);
			for (unsigned int row = 0; row < 3; ++row)
			{
				for (unsigned int col = 0; col < 3; ++col)
				{
					view_matrix_.at<float>(row, col) = (float)rot.at<double>(row, col);
				}
				view_matrix_.at<float>(row, 3) = (float)tvecs[i][row];
			}
			view_matrix_.at<float>(3, 3) = 1.0f;

			//cv to gl
			cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
			cvToGl.at<float>(0, 0) = 1.0f;
			cvToGl.at<float>(1, 1) = -1.0f;
			cvToGl.at<float>(2, 2) = -1.0f;
			cvToGl.at<float>(3, 3) = 1.0f;
			view_matrix_ = cvToGl * view_matrix_;
			cv::transpose(view_matrix_, view_matrix_);
		}
		return true;
	}

	return false;
}

void Camera::markerless_init(string src_path, int minHessian)
{
	this->img_object = imread(src_path, IMREAD_GRAYSCALE);
	this->detector = xfeatures2d::SURF::create(minHessian);
	this->detector->detectAndCompute(this->img_object, noArray(), this->keypoints_object, this->descriptors_object);
}

bool Camera::marker_less_compute(Mat & frame)
{
	Mat img = frame.clone();
	cvtColor(img, img, COLOR_BGR2GRAY);
	//计算当前帧的特征
	std::vector<KeyPoint> keypoints_scene;
	Mat descriptors_scene;
	this->detector->detectAndCompute(img, noArray(), keypoints_scene, descriptors_scene);

	if (keypoints_scene.size() <= 0)
		return false;
	//特征匹配
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	std::vector< std::vector<DMatch> > knn_matches;

	matcher->knnMatch(descriptors_object, descriptors_scene, knn_matches, 2);

	//精简特征
	const float ratio_thresh = 0.75f;
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	if (good_matches.size() > 10)
	{
		cv::Mat rot;
		//计算单应性矩阵
		std::vector<Point2f> obj;
		std::vector<Point2f> scene;

		for (size_t i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
			scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
		}

		Mat H = findHomography(obj, scene, RANSAC);

		std::vector<Point2f> obj_corners(4);
		obj_corners[0] = Point2f(0, 0);
		obj_corners[1] = Point2f((float)img_object.cols, 0);
		obj_corners[2] = Point2f((float)img_object.cols, (float)img_object.rows);
		obj_corners[3] = Point2f(0, (float)img_object.rows);

		//应用单应性矩阵
		std::vector<Point2f> scene_corners(4);
		perspectiveTransform(obj_corners, scene_corners, H);

		std::vector<Point3f> obj_corners_3d(4);
		for (size_t i = 0; i < 4; i++)
			obj_corners_3d[i] = Point3f(obj_corners[i].x / 640.f - 0.5, -obj_corners[i].y / 640.f + 0.5, 0);
		//相机姿态估计
		cv::Vec3d rvec, tvec;
		cv::solvePnP(obj_corners_3d, scene_corners, camera_matrix_, dist_coeffs_, rvec, tvec);

		//以下同maker-based 相同
		view_matrix_ = cv::Mat::zeros(4, 4, CV_32F);

		Rodrigues(rvec, rot);
		for (unsigned int row = 0; row < 3; ++row)
		{
			for (unsigned int col = 0; col < 3; ++col)
			{
				view_matrix_.at<float>(row, col) = (float)rot.at<double>(row, col);
			}
			view_matrix_.at<float>(row, 3) = (float)tvec[row];
		}
		view_matrix_.at<float>(3, 3) = 1.0f;

		cv::Mat cvToGl = cv::Mat::zeros(4, 4, CV_32F);
		cvToGl.at<float>(0, 0) = 1.0f;
		cvToGl.at<float>(1, 1) = -1.0f; // Invert the y axis 
		cvToGl.at<float>(2, 2) = -1.0f; // invert the z axis 
		cvToGl.at<float>(3, 3) = 1.0f;
		view_matrix_ = cvToGl * view_matrix_;
		cv::transpose(view_matrix_, view_matrix_);

		return true;
	}
	return false;
}

int Camera::getWidth()
{
	return fwidth_;
}

int Camera::getHeight()
{
	return fheight_;
}

mat4 Camera::get_view_matrix()
{
	glm::mat4 temp;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			temp[i][j] = view_matrix_.at<float>(i, j);

	return temp;
}

mat4 Camera::get_projection_matrix()
{
	glm::mat4 temp;
	for (int i = 0; i < 4; i++)
		for (int j = 0; j < 4; j++)
			temp[i][j] = projection_matrix_.at<float>(i, j);

	return temp;
}
