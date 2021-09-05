#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <fstream>
#include <string.h>
#include <sstream>
#include <vector>
#include<iostream>
#include<time.h>

//打开文件夹
#if defined(__linux__)
#include "include/GetFileName_linux.h"
#include <unistd.h>  
#include <sys/types.h>  
#include <sys/stat.h>  
#elif defined(_WIN32)
#include "GetFileName.h"
#include <direct.h>
#endif

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

#define DEBUG false

void mkdir(const string& path){
#if defined(__linux__)
	if (access(path.c_str(), 0) == -1){
		mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
	}
#elif defined(_WIN32)
	if (_access(path.c_str(), 0) == -1){
		_mkdir(path.c_str());
	}
#endif
}

class InputType{
public:
	vector<KeyPoint> vkp1, vkp2;
	Size img1shape, img2shape;
	vector<DMatch> vDMatches;
	InputType(){

	}
	InputType(const vector<KeyPoint> &vkp1, Size img1shape, const vector<KeyPoint> &vkp2, Size img2shape, const vector<DMatch> &vDMatches){
		this->vkp1 = vkp1;
		this->img1shape = img1shape;
		this->vkp2 = vkp2;
		this->img2shape = img2shape;
		this->vDMatches = vDMatches;
	}
	InputType(const InputType& obj){
		this->vkp1 = obj.vkp1;
		this->img1shape = obj.img1shape;
		this->vkp2 = obj.vkp2;
		this->img2shape = obj.img2shape;
		this->vDMatches = obj.vDMatches;
	}
	~InputType(){

	}
};

enum Feature_type { GCC_SIFT, GCC_SURF, GCC_ORB };

struct MethodPara{
	float sigma;
	int neiborwidth;
	float threshold;
	MethodPara(){
		this->sigma = 0;
		this->neiborwidth = 1;
		this->threshold = 2;
	}
	MethodPara(float sigma,
	int neiborwidth,
	float threshold){
		this->sigma = sigma;
		this->neiborwidth = neiborwidth;
		this->threshold = threshold;
	}
};

struct MatchPara{
	string method;
	int feature_number;
	MethodPara mp;
	Feature_type feature_type;
	MatchPara(string method,
		float sigma,
		int neiborwidth,
		float threshold,
		int feature_number=10000,
		Feature_type ft=GCC_ORB){
		this->method = method;
		this->feature_number = feature_number;
		this->mp = MethodPara(sigma,neiborwidth,threshold);
		this->feature_type = ft;
	}
};

struct Parameter{
	void* data;
	string name;
	Parameter(void* data, string name){
		this->data = data;
		this->name = name;
	}
};

class readParaPointer{
	vector<Parameter> para;
public:
	readParaPointer(char* parafilepath,bool showpara=true,bool showcomments=true){
		ifstream f;
		f.open(parafilepath);
		string s;
		cout << "readparameter:" << endl;
		while (getline(f, s))
		{
			if (!s.empty())
			{
				//跳过注释项
				if (s[0] == '#'){
					if (showcomments)
						cout << s << endl;
					continue;
				}
				stringstream ss;
				ss << s;
				string name;
				char type;
				void* data = NULL;
				ss >> name;
				if (showpara) cout << "\t" << name << "=";
				ss >> type;
				switch (type){
				case 's':
					data = new string();
					ss >> *(string*)data;
					if (showpara) cout << *(string*)data << endl;
					break;
				case 'i':
					data = new int();
					ss >> *(int*)data;
					if (showpara) cout << *(int*)data << endl;
					break;
				case 'f':
					data = new float();
					ss >> *(float*)data;
					if (showpara) cout << *(float*)data << endl;
					break;
				case 'd':
					data = new double();
					ss >> *(double*)data;
					if (showpara) cout << *(double*)data << endl;
					break;
				default:
					cerr << " unknow data type:" << s << endl;
					break;
				}
				
				Parameter p(data, name);
				para.push_back(p);
			}

		}
		f.close();
	}
	~readParaPointer(){

	}
	void * getvalue(char* name){
		for (int i = 0; i < para.size(); i++){
			if (strcmp(name, para[i].name.c_str()) == 0){
				return para[i].data;
			}
		}
		cerr << "no para name '" << name << "'" << endl;
		return 0;
	}
};

class NetFunc{
public:
	static Mat sum_Pooling(Mat input,Size win_size){
		int ph = win_size.height;
		int pw = win_size.width;
		int h = (input.rows-1) / ph+1;
		int w = (input.cols - 1) / pw + 1;
		Mat results = Mat::zeros(Size(w, h), CV_32FC1);
		for (int i = 0; i < h; i++){
			int rh = ph;
			if ((i + 1)*ph>input.rows){
				rh = input.rows - i*ph;
			}
			for (int j = 0; j < w; j++){
				int rw = pw;
				if ((j + 1)*pw>input.cols){
					rw = input.cols - j*pw;
				}
				Rect rect(j*pw, i*ph, rw, rh);
				results.at<float>(i, j) = sum(input(rect))[0];
			}
		}
		return results;
	}
	//static void CambriconSimulation
};


float guassFun(float sigma, float dist){
	if (sigma == 0){
		return 0;
	}
	return exp(-pow(dist / sigma, 2));
}

InputType getSIFTInput(const Mat& img1, const Mat& img2, int feature_number = 10000){

	vector<KeyPoint> kp1, kp2;
	vector<DMatch> vMatchesAll;
	//int resolution = img1.rows * img1.cols;
	//int feature_number = 10000;
	clock_t bg, ed;
	//fast特征点检测
	Ptr<ORB> IPextractor = ORB::create(feature_number);
	if (img1.cols*img1.rows <= 640 * 480){
		IPextractor->setFastThreshold(0);
	}
	IPextractor->detect(img1, kp1);
	IPextractor->detect(img2, kp2);
	//SIFT计算
	Ptr<SIFT> extractor = SIFT::create();
	Mat d1, d2;
	bg = clock();
	extractor->compute(img1, kp1, d1);
	extractor->compute(img2, kp2, d2);
	ed = clock();
	if (DEBUG)
		cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;

#ifdef USE_GPU //使用GPU
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat descriptors_gpu_src(d1), descriptors_gpu_dst(d2);
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
#else//使用CPU
	/*
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	*/
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher;
	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,matches_all);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
#endif
	/*
	#ifdef USE_GPU //使用GPU
	clock_t bg, ed;
	cuda::GpuMat src_gpu(img1);
	cuda::GpuMat dst_gpu(img2);
	cuda::GpuMat descriptors_gpu_src, descriptors_gpu_dst;
	Ptr<cuda::ORB> extractor = cuda::ORB::create();
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat fullmask_1(src_gpu.size(), CV_8U, 0xFF);
	cuda::GpuMat fullmask_2(dst_gpu.size(), CV_8U, 0xFF);
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	bg = clock();
	extractor->detectAndCompute(src_gpu, fullmask_1, kp1, descriptors_gpu_src);
	extractor->detectAndCompute(dst_gpu, fullmask_2, kp2, descriptors_gpu_dst);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) *1000 << "ms" << endl;
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
	#else//使用CPU
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING);
	// extract features
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg)/CLOCKS_PER_SEC)*1000 << "ms" << endl;

	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,vMatchesAll);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
	#endif
	*/
	if (DEBUG)
		cout << "Nearest neighbor matching time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;
	return InputType(kp1, Size(img1.cols, img1.rows), kp2, Size(img2.cols, img2.rows), vMatchesAll);
}

InputType getSURFInput(const Mat& img1, const Mat& img2, int feature_number = 10000){

	vector<KeyPoint> kp1, kp2;
	vector<DMatch> vMatchesAll;
	//int resolution = img1.rows * img1.cols;
	//int feature_number = 10000;
	clock_t bg, ed;
	Ptr<SURF> extractor = SURF::create();
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
		cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;

#ifdef USE_GPU //使用GPU
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat descriptors_gpu_src(d1), descriptors_gpu_dst(d2);
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
#else//使用CPU
	/*
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	*/
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher;
	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,matches_all);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
#endif
	/*
	#ifdef USE_GPU //使用GPU
	clock_t bg, ed;
	cuda::GpuMat src_gpu(img1);
	cuda::GpuMat dst_gpu(img2);
	cuda::GpuMat descriptors_gpu_src, descriptors_gpu_dst;
	Ptr<cuda::ORB> extractor = cuda::ORB::create();
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat fullmask_1(src_gpu.size(), CV_8U, 0xFF);
	cuda::GpuMat fullmask_2(dst_gpu.size(), CV_8U, 0xFF);
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	bg = clock();
	extractor->detectAndCompute(src_gpu, fullmask_1, kp1, descriptors_gpu_src);
	extractor->detectAndCompute(dst_gpu, fullmask_2, kp2, descriptors_gpu_dst);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) *1000 << "ms" << endl;
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
	#else//使用CPU
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING);
	// extract features
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg)/CLOCKS_PER_SEC)*1000 << "ms" << endl;

	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,vMatchesAll);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
	#endif
	*/
	if (DEBUG)
		cout << "Nearest neighbor matching time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;
	return InputType(kp1, Size(img1.cols, img1.rows), kp2, Size(img2.cols, img2.rows), vMatchesAll);
}

InputType getORBInput(const Mat& img1, const Mat& img2, int feature_number = 10000){

	vector<KeyPoint> kp1, kp2;
	vector<DMatch> vMatchesAll;
	int resolution = img1.rows * img1.cols;
	//int feature_number = 10000;
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	extractor->setMaxFeatures(feature_number);
	if (resolution <= 480 * 640){
		extractor->setFastThreshold(0);
	}
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
		cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;

#ifdef USE_GPU //使用GPU
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat descriptors_gpu_src(d1), descriptors_gpu_dst(d2);
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
#else//使用CPU
	/*
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	*/
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING);
	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,matches_all);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
#endif
	/*
	#ifdef USE_GPU //使用GPU
	clock_t bg, ed;
	cuda::GpuMat src_gpu(img1);
	cuda::GpuMat dst_gpu(img2);
	cuda::GpuMat descriptors_gpu_src, descriptors_gpu_dst;
	Ptr<cuda::ORB> extractor = cuda::ORB::create();
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat fullmask_1(src_gpu.size(), CV_8U, 0xFF);
	cuda::GpuMat fullmask_2(dst_gpu.size(), CV_8U, 0xFF);
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	bg = clock();
	extractor->detectAndCompute(src_gpu, fullmask_1, kp1, descriptors_gpu_src);
	extractor->detectAndCompute(dst_gpu, fullmask_2, kp2, descriptors_gpu_dst);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) *1000 << "ms" << endl;
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
	#else//使用CPU
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING);
	// extract features
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg)/CLOCKS_PER_SEC)*1000 << "ms" << endl;

	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,vMatchesAll);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
	#endif
	*/
	if (DEBUG)
		cout << "Nearest neighbor matching time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;
	return InputType(kp1, Size(img1.cols, img1.rows), kp2, Size(img2.cols, img2.rows), vMatchesAll);
}

InputType getORBWithRatio(const Mat& img1, const Mat& img2, int feature_number = 10000){

	vector<KeyPoint> kp1, kp2;
	vector<DMatch> vMatchesAll;
	int resolution = img1.rows * img1.cols;
	//int feature_number = 10000;
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	extractor->setMaxFeatures(feature_number);
	if (resolution <= 480 * 640){
		extractor->setFastThreshold(0);
	}
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
		cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;

#ifdef USE_GPU //使用GPU
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat descriptors_gpu_src(d1), descriptors_gpu_dst(d2);
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
#else//使用CPU
	/*
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	*/
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING2);//不适用交叉验证，以其返回多个匹配用作ratio排除
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints,AllMatchPoints;

	vector<Mat> train_desc(1, d1);
	nnmatcher.add(train_desc);
	nnmatcher.train();
	nnmatcher.knnMatch(d2, matchePoints, 2);
	cout << "total match points: " << matchePoints.size() << endl;
	// Lowe's algorithm,获取优秀匹配点
	float ratio = 0.66;
	for (int i = 0; i < matchePoints.size(); i++)
	{
		AllMatchPoints.push_back(matchePoints[i][0]);
		if (matchePoints[i][0].distance <ratio * matchePoints[i][1].distance)
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}
	// nearest-neighbor matching
	//Ptr<FeatureDetector> FeatureDetector=Feature;
	//DescriptorExtractor;
	//Feature2D;
	//DescriptorExtractor;
	//DescriptorMatcher;
	//initModule_nonfree();//初始化模块，使用SIFT或SURF时用到

	//nnmatcher->match(d1,d2,matches_all);
	/*
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
	cout << "good match points: " << GoodMatchePoints.size() << endl;
	cout << "All match points: " << AllMatchPoints.size() << endl;
	cout << "Pu match points: " << vMatchesAll.size() << endl;
	Mat Good, All, Pu;
	drawMatches(img1,kp1,img2,kp2,GoodMatchePoints,Good);
	drawMatches(img1, kp1, img2, kp2, AllMatchPoints, All);
	drawMatches(img1, kp1, img2, kp2, vMatchesAll, Pu);
	imshow("Good", Good);
	imshow("ALl", All);
	imshow("Pu", Pu);
	waitKey(0);*/
#endif
	/*
	#ifdef USE_GPU //使用GPU
	clock_t bg, ed;
	cuda::GpuMat src_gpu(img1);
	cuda::GpuMat dst_gpu(img2);
	cuda::GpuMat descriptors_gpu_src, descriptors_gpu_dst;
	Ptr<cuda::ORB> extractor = cuda::ORB::create();
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	cuda::GpuMat fullmask_1(src_gpu.size(), CV_8U, 0xFF);
	cuda::GpuMat fullmask_2(dst_gpu.size(), CV_8U, 0xFF);
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	bg = clock();
	extractor->detectAndCompute(src_gpu, fullmask_1, kp1, descriptors_gpu_src);
	extractor->detectAndCompute(dst_gpu, fullmask_2, kp2, descriptors_gpu_dst);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) *1000 << "ms" << endl;
	// nearest-neighbor matching
	bg = clock();
	matcher->match(descriptors_gpu_src, descriptors_gpu_dst, vMatchesAll);
	ed = clock();
	#else//使用CPU
	clock_t bg, ed;
	Ptr<ORB> extractor = ORB::create();
	if (resolution <= 480 * 640){
	extractor->setMaxFeatures(feature_number);
	extractor->setFastThreshold(0);
	}
	else
	{
	extractor->setMaxFeatures(feature_number * 10);
	extractor->setFastThreshold(5);
	}
	//Ptr<BFMatcher> nnmatcher = BFMatcher::create(NORM_HAMMING);
	BFMatcher nnmatcher(NORM_HAMMING);
	// extract features
	Mat d1, d2;
	bg = clock();
	extractor->detectAndCompute(img1, Mat(), kp1, d1);
	extractor->detectAndCompute(img2, Mat(), kp2, d2);
	ed = clock();
	if (DEBUG)
	cout << "Feature extraction time consuming: " << (double(ed - bg)/CLOCKS_PER_SEC)*1000 << "ms" << endl;

	// nearest-neighbor matching

	//nnmatcher->match(d1,d2,vMatchesAll);
	bg = clock();
	nnmatcher.match(d1, d2, vMatchesAll);
	ed = clock();
	#endif
	*/
	if (DEBUG)
		cout << "Nearest neighbor matching time consuming: " << (double(ed - bg) / CLOCKS_PER_SEC) * 1000 << "ms" << endl;
	return InputType(kp1, Size(img1.cols, img1.rows), kp2, Size(img2.cols, img2.rows), AllMatchPoints);
}

inline Mat DrawInlier(Mat &src1, Mat &src2, vector<KeyPoint> &kpt1, vector<KeyPoint> &kpt2, vector<DMatch> &inlier, Scalar linecolor = Scalar::all(255), int type = 1, Scalar pointcolor = Scalar::all(0)) {
	const int height = max(src1.rows, src2.rows);
	const int width = src1.cols + src2.cols;
	Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
	src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
	src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

	if (type == 1)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, linecolor,1.5);
		}
	}
	else if (type == 2)
	{
		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			line(output, left, right, linecolor,1.5);
		}

		for (size_t i = 0; i < inlier.size(); i++)
		{
			Point2f left = kpt1[inlier[i].queryIdx].pt;
			Point2f right = (kpt2[inlier[i].trainIdx].pt + Point2f((float)src1.cols, 0.f));
			circle(output, left, 1, pointcolor, 2);
			circle(output, right, 1, pointcolor, 2);
		}
	}

	return output;
}
inline void imresize(Mat &src, int height) {
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}