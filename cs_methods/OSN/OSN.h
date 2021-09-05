#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <ctime>
//#include "base.h"
#define PI 3.1415926535897932384
using namespace std;
using namespace cv;
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

class OSN
{
private:

	//A，B坐标    (372,40)
	vector<Point2i> mvP1, mvP2;
	// 五千 mvp12序号对(0,769) (1,1856)-----(4999,2331)

	vector<pair<int, int> > mvMatches;

	//圆心数量&坐标
	int NumberCircleLeft, NumberCircleRight;
	vector<Point2i> LeftCircleCenter;
	vector<Point2i> RightCircleCenter;
	Size sizeleft, sizeright;
	//五千对点的圆心序号
	Mat PerCellLeftCenter;
	Mat PerCellRightCenter;
	Mat AllPerCellLeftCenter;
	Mat AllPerCellRightCenter;
	//原数组A&B
	//vector<Mat> A;
	Mat B;
	vector<pair<int, int>> *LeftCircleMatchIdx;

	//init para
	float R;           //圆形区域半径
	int m = 400;//s设置400个固定圆心
	int neighbor = 4;     //kernel相关
	const float msigma = 1.0;
	const float mthreshold = 2.0;
	int K = 5;           //圆层数
	//4次移动
	int h_ = 0, w_ = 0;

	// Number of Matches
	size_t mNumberMatches;

	//矩阵X
	Mat mMotionStatistics;

	//数组N
	vector<int> mNumberPointsInPerCellLeft;
	//数组C
	vector<int> mCellPairs;

	// Inlier Mask for output
	vector<bool> mvbInlierMask;

	Mat kernel;
	float sigma;
	float thresh_factor;
	int neighborwidth;


public:
	OSN(const vector<KeyPoint> &vkp1, const Size size1, const vector<KeyPoint> &vkp2, const Size size2, const vector<DMatch> &vDMatches, MethodPara mp = MethodPara(1.75, 4, 2.0),int fixedC=400)
	{
		NormalizePoints(vkp1, size1, mvP1);
		NormalizePoints(vkp2, size2, mvP2);
		mNumberMatches = vDMatches.size();
		ConvertMatches(vDMatches, mvMatches);

		sigma = mp.sigma;
		thresh_factor = mp.threshold;
		neighbor = mp.neiborwidth;
		K = neighbor + 1;
		neighborwidth = 2 * neighbor + 1;
		kernel = getKernel(sigma);
		
		//通过圆心数量m计算R的半径
		this->m = fixedC;
		this->R = ceil(sqrt(float(size1.width*size1.height) / (PI*m)));
		
		//sizeleft = Size(size1.width / R - 1, size1.height / R - 1);
		//sizeright = Size(size2.width / R - 1, size2.height / R - 1);
		sizeleft.width = ceil(size1.width / (2 * R));
		sizeleft.height = ceil(size1.height / (2 * R));
		sizeright.width = ceil(size2.width / (2 * R));
		sizeright.height = ceil(size2.height / (2 * R));

		NumberCircleLeft = sizeleft.width*sizeleft.height;
		NumberCircleRight = sizeright.width*sizeright.height;

		//得到所有圆心坐标
		for (int i = 1; i <= sizeleft.height; i++)
		for (int j = 1; j <= sizeleft.width; j++)
			LeftCircleCenter.push_back(Point2i((2 * j - 1)*R, (2 * i - 1)*R));
		for (int i = 1; i <= sizeright.height; i++)
		for (int j = 1; j <= sizeright.width; j++)
			RightCircleCenter.push_back(Point2i((2 * j - 1)*R, (2 * i - 1)*R));
		PerCellLeftCenter = Mat::zeros(mvP1.size(), 1, CV_32SC1);
		PerCellRightCenter = Mat::zeros(mvP2.size(), 1, CV_32SC1);
		/*
		//得到所有点附近圆心序号
		AllPerCellLeftCenter = InitCenter(PerCellLeftCenter, mvP1, LeftCircleCenter, sizeleft);
		AllPerCellRightCenter = InitCenter(PerCellRightCenter, mvP2, RightCircleCenter, sizeright);
		*/
		//for (int j = 0; j < 200; j++) {
		//	cout << "n=" << j<<"  ";
		//	for (int k = 0; k < 4; k++) {
		//		int tmp= PerCellRightCenter.at<int>(j,k);
		//		cout << tmp << " ";
		//	}
		//	cout << endl;
		//}
	};
	~OSN() {};

	// Get Inlier Mask
	// Return number of inliers 
	int GetInlierMask(vector<bool> &vbInliers) {
		int max_inlier = run();
		vbInliers = mvbInlierMask;
		return max_inlier;
	}

private:
	Mat getKernel(double sigma) {
		Mat k;
		//Mat r = getGaussianKernel(K * 2 + 1, sigma);
		//Mat c = getGaussianKernel(K * 2 + 1, sigma).reshape(1, 1);
		//k = (r*c);
		//k.convertTo(k, CV_32FC1);
		if (sigma == 0){
			k = Mat::ones(Size(K, 1), CV_32FC1);
		}
		else{/*
			 k = Mat::ones(Size(K, 1), CV_32FC1);
			 float value = 1 / float(K);
			 for (int i = 0; i < K; i++){
			 k.at<float>(0, i) = 1 - value*i;
			 }*/
			Mat tmp = getGaussianKernel(K * 2, sigma).reshape(1, 1)(Rect(K, 0, K, 1));
			k = tmp * 2;
			k.convertTo(k, CV_32FC1);
		}
		return k;
	}

	// Normalize Key Points to Range(0 - 1)
	void NormalizePoints(const vector<KeyPoint> &kp, const Size &size, vector<Point2i> &npts) {
		const size_t numP = kp.size();
		npts.resize(numP);

		for (size_t i = 0; i < numP; i++)
		{
			npts[i].x = kp[i].pt.x;
			npts[i].y = kp[i].pt.y;
		}
	}

	// Convert OpenCV DMatch to Match (pair<int, int>)
	void ConvertMatches(const vector<DMatch> &vDMatches, vector<pair<int, int> > &vMatches) {
		vMatches.resize(mNumberMatches);
		for (size_t i = 0; i < mNumberMatches; i++)
		{
			vMatches[i] = pair<int, int>(vDMatches[i].queryIdx, vDMatches[i].trainIdx);
		}
	}
	int run() {
		mvbInlierMask.assign(mNumberMatches, false);
		mMotionStatistics = Mat::zeros(NumberCircleLeft, NumberCircleRight, CV_32SC1);
		//移动四次点阵，获取所有点
		for (int dotType = 0; dotType < 4; dotType++){
			init(dotType);
			mMotionStatistics.setTo(0);
			B = Mat::zeros(NumberCircleLeft, K, CV_32SC1);
			LeftCircleMatchIdx = new vector<pair<int, int>>[NumberCircleLeft];
			mCellPairs.assign(NumberCircleLeft, -1);
			mNumberPointsInPerCellLeft.assign(NumberCircleLeft, 0);
			AssignMatchPairs();
			VerifyCellPairs();
			delete[] LeftCircleMatchIdx;
			for (size_t i = 0; i < mNumberMatches; i++)
			{
				//这里有错误，因为不是匹配对
				int lpn = mvMatches[i].first;
				int rpn = mvMatches[i].second;
				int first = AllPerCellLeftCenter.at<int>(lpn, 0);
				int second = AllPerCellRightCenter.at<int>(rpn, 0);
				if (second != -1 && mCellPairs[first] == second)
				{
					//如果左点圆心最匹配的右点圆心序号==右点圆心序号，此匹配被接受
					mvbInlierMask[i] = true;
				}
			}
		}
		int num_inlier = sum(mvbInlierMask)[0];
		return num_inlier;
	}
	//计算数组N(mNumberPointsInPerCellLeft),矩阵X(mMotionStatistics)
	void AssignMatchPairs() {
		for (size_t i = 0; i < mNumberMatches; i++) {
			//这里有错误，因为不是匹配对
			int lpn = mvMatches[i].first;//左点序号
			int rpn = mvMatches[i].second;//右点序号
			int LC_idx = PerCellLeftCenter.at<int>(lpn, 0);    //左点圆心序号
			int RC_idx = PerCellRightCenter.at<int>(rpn, 0);   //右点圆心序号
			if (LC_idx != -1 && RC_idx != -1){
				mNumberPointsInPerCellLeft[LC_idx]++;
				mMotionStatistics.at<int>(LC_idx, RC_idx)++;
			}
			//数据预处理------------------------------------------------------------------------
			Point2i &p = mvP1[lpn];
			//移动网格
			//Point2i q = Point2i((p.x + w_) / (2 * R), (p.y + h_) / (2 * R));
			Point2i q = Point2i(p.x  / (2 * R), p.y  / (2 * R));
			//int id = AllPerCellLeftCenter.at<int>(i,0);
			//Point2i q = Point2i(id%sizeleft.width, id / sizeleft.width);
			//Point2i q = Point2i(LC_idx%sizeleft.width, LC_idx / sizeleft.width);
			Point2i TempUp = q;
			Point2i TempDown = q;
			while (TempUp.y != -1 && getLevel(LeftCircleCenter[getidx(TempUp, sizeleft)], p) < K) {
				Point2i TempLeft = TempUp;
				Point2i TempRight = TempUp;
				while (TempLeft.x != -1 && getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p) < K) {
					LeftCircleMatchIdx[getidx(TempLeft, sizeleft)].push_back(pair<int, int>(i, getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p)));
					B.at<int>(getidx(TempLeft, sizeleft), getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p))++;
					TempLeft.x--;
				}
				TempRight.x++;
				while (TempRight.x != sizeleft.width && getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p) < K) {
					LeftCircleMatchIdx[getidx(TempRight, sizeleft)].push_back(pair<int, int>(i, getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p)));
					B.at<int>(getidx(TempRight, sizeleft), getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p))++;
					TempRight.x++;
				}
				TempUp.y--;
			}
			TempDown.y++;
			while (TempDown.y != sizeleft.height && getLevel(LeftCircleCenter[getidx(TempDown, sizeleft)], p) < K) {
				Point2i TempLeft = TempDown;
				Point2i TempRight = TempDown;
				while (TempLeft.x != -1 && getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p) < K) {
					LeftCircleMatchIdx[getidx(TempLeft, sizeleft)].push_back(pair<int, int>(i, getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p)));
					B.at<int>(getidx(TempLeft, sizeleft), getLevel(LeftCircleCenter[getidx(TempLeft, sizeleft)], p))++;
					TempLeft.x--;
				}
				TempRight.x++;
				while (TempRight.x != sizeleft.width && getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p) < K) {
					LeftCircleMatchIdx[getidx(TempRight, sizeleft)].push_back(pair<int, int>(i, getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p)));
					B.at<int>(getidx(TempRight, sizeleft), getLevel(LeftCircleCenter[getidx(TempRight, sizeleft)], p))++;
					TempRight.x++;
				}
				TempDown.y++;
			}
		}

		//for (int n = 0; n < NumberCircleLeft; n++) {
		//	cout << "n=" << n << " ";
		//	for (int k = 0; k < K; k++) {
		//		cout << B.at<int>(n, k) << " ";
		//	}
		//	//if (sum(B.row(n))[0] == int(LeftCircleMatchIdx[n].size()))
		//	//	cout << "right";
		//	//else
		//	//	cout << "wrong";
		//	cout << endl;
		//}

		//for (int n = 0; n < NumberCircleLeft; n++) {
		//	cout << "n=" << n << " ";
		//	for (int k = 0; k <LeftCircleMatchIdx[n].size() ; k++) {
		//		cout << LeftCircleMatchIdx[n][k] << " ";
		//	}
		//	cout << endl;
		//}
	}

	void VerifyCellPairs() {
		int inlinesnum = 0;
		for (int i = 0; i < NumberCircleLeft; i++) {
			if (sum(mMotionStatistics.row(i))[0] == 0) {
				mCellPairs[i] = -1;
				continue;
			}
			//找最匹配圆
			int max_number = 0;
			for (int j = 0; j < NumberCircleRight; j++) {
				int value = mMotionStatistics.at<int>(i, j);
				if (value > max_number)
				{
					mCellPairs[i] = j;
					max_number = value;
				}
			}
			//最匹配圆心序号left:i right:idx_circle_rt,圆心坐标LeftCircleCenter[i],RightCircleCenter[idx_circle_rt]
			int idx_circle_rt = mCellPairs[i];
			//A[] 左圆与右圆在相同层数上的匹配数
			//B[] 左圆特征总数

			vector<int> A;
			A.assign(K, 0);
			for (int n = 0; n < LeftCircleMatchIdx[i].size(); n++) {
				int R1 = LeftCircleMatchIdx[i][n].second;
				int R2 = getLevel(RightCircleCenter[idx_circle_rt], mvP2[mvMatches[LeftCircleMatchIdx[i][n].first].second]);
				if (R1 < K&&R1 == R2) {
					A[R1]++;
				}
			}
			float score = 0;
			float thresh = 0;
			//vector<int> A, B;
			//A.assign(K, 0);
			//B.assign(K, 0);
			//for (size_t n = 0; n < mNumberMatches; n++) {
			//	int R1 = getLevel(LeftCircleCenter[i], mvP1[mvMatches[n].first]);
			//	int R2 = getLevel(RightCircleCenter[idx_circle_rt], mvP2[mvMatches[n].second]);
			//	if (R1 < K&&R1 == R2) {
			//		A[R1]++;
			//	}
			//	if (R1 < K)
			//		B[R1]++;
			//}
			//if (A[0] < mMotionStatistics.at<int>(i, idx_circle_rt)){
			//	cout << "error A[0] must bigger than maxnum(A[0]=" << A[0] << ",maxnum=" << mMotionStatistics.at<int>(i, idx_circle_rt) << ")" << endl;
			//}
			//cout << "A[]= ";
//			cout << "n=" << i << " ";
			for (int m = 0; m < K; m++) {
				//float kernelValue = kernel.at<float>(K, K + m);
				float kernelValue = kernel.at<float>(0, m);
//				cout << A[m]<<" ";
				score += kernelValue*A[m];
				//int tmp = 64;
				//if (m == 0)
				//	tmp = 1;
				//thresh += tmp*kernelValue*kernelValue*B[m];
				if (m == 0)
					thresh += kernelValue*kernelValue*B.at<int>(i,m);
				else
					//thresh += 8 * m*kernelValue*kernelValue*B[m];
					thresh += kernelValue*kernelValue*B.at<int>(i, m);
			}
//			cout << endl;
			
			//cout << "n=" << i << " ";
			//for (int m = 0; m < K; m++) {
			//	cout << B.at<int>(i,m) << " ";
			//	//cout << A[m] << " ";
			//}
			//cout << endl;
			thresh = thresh_factor*sqrt(thresh);
			if (score <= thresh)
				mCellPairs[i] = -2;
			else{
				inlinesnum += A[0];
				//inlinesnum += A[i].at<int>(idx_circle_rt, 0);
				//cout << "s=" << score << ">t=" << thresh <<",inlinesnum="<<inlinesnum<< endl;
			}
		}
	}

	void init(int dotType){
		switch (dotType)
		{
		case 0:break;
		case 1:
			h_ = R;
			break;
		case 2:
			w_ = R;
			break;
		case 3:
			h_ = R;
			w_ = R;
			break;
		default:
			break;
		}
		AllPerCellLeftCenter = InitCenter(PerCellLeftCenter, mvP1, LeftCircleCenter, sizeleft);
		AllPerCellRightCenter = InitCenter(PerCellRightCenter, mvP2, RightCircleCenter, sizeright);
	}

	Mat InitCenter(Mat &mat, vector<Point2i> mvP, vector<Point2i> CircleCenter, const Size size) {
		Mat AllPerCellCenter = Mat::zeros(mvP.size(), 1, CV_32SC1);
		for (int n = 0; n < mvP.size(); n++) {
			Point2i &p = mvP[n];
			//移动网格
			Point2i q = Point2i((p.x + w_) / (2 * R), (p.y + h_) / (2 * R));
			//int id = q.y*size.width + q.x;
			int id = getidx(q, size);
			AllPerCellCenter.at<int>(n, 0) = id;
			if (id >= 0 && id<CircleCenter.size() && getLevel(CircleCenter[id], p) == 0)
				mat.at<int>(n, 0) = id;
			else{
				mat.at<int>(n, 0) = -1;
			}
		}
		return AllPerCellCenter;
	}
	//得到圆心序号 i为列，j为行
	inline int getidx(Point2i q, Size size) {
		return q.y*size.width + q.x;
	}
	//得到两点距离的平方
	inline int distance(Point2i A, Point2i B) {
		return (A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y);
	}
	//得到点距离圆心的层数
	inline int getLevel(Point2i center, Point2i p) {
		//		int level = floor(distance(center, p) / (Rin*Rin));       //相同环面积
		p = Point2i(p.x + w_, p.y + h_);
		float dist = sqrt(distance(center, p));
		int level = floor(dist / R);       //相同环间隔
		//level = (level + 1) / 2;//相同直径
		//int level = floor(distance(center, p) / (Rin*Rin));       //相同环面积
		return level;
	}
};