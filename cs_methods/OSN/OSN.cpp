#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <vector>
// #include "base.h"
#include "OSN.h"
using namespace std;
using namespace cv;

//DMatch
vector<DMatch> get_corres(char* corres_p){
    // read text
    vector<DMatch> corres;
    ifstream f;
    f.open(corres_p);
    int qi=0, ti=0;
    while(!f.eof()) //直到文件结尾
	{
		f>>ti;
// 		cout<<qi<<","<<ti<<endl;
        corres.push_back(DMatch(qi,ti,0));
        qi+=1;
	} 
    f.close();
    corres.pop_back();
    return corres;
}
Size get_size(char* imgsize_p){
    
    ifstream f;
    int a=0,b=0;
    f.open(imgsize_p);
    f>>a>>b;
    f.close();
    return Size(a,b);
}
//KeyPoints
vector<KeyPoint> get_kps(char* kp_p){
    // read text
    vector<KeyPoint> kp;
    ifstream f;
    f.open(kp_p);
    float a=0,b=0;
    while(!f.eof()) //直到文件结尾
	{
		f>>a>>b;
		//cout<<a<<","<<b<<endl;
        kp.push_back(KeyPoint(a,b,0));
	} 
    f.close();
    kp.pop_back();
    return kp;
}
int main(int argc, char **argv){
    // get text file
    if (argc != 4)
	{
		cerr << endl << "Usage: ./OSN Kp1 Img1Size Kp2 Img2Size Correspodences1to2 ("<<argc<<")\n" << endl;
    }
    char* kp1_p=argv[1];
    char* img1size_p=argv[2];
    char* kp2_p=argv[3];
    char* img2size_p=argv[4];
    char* corres_p=argv[5];
    cout<<kp1_p<<","<<img1size_p<<","<<kp2_p<<","<<img2size_p<<","<<corres_p<<endl;
    vector<KeyPoint> kp1=get_kps(kp1_p);
    Size img1size=get_size(img1size_p);
    vector<KeyPoint> kp2=get_kps(kp2_p);
    Size img2size=get_size(img2size_p);
    vector<DMatch> corres=get_corres(corres_p);
    // Employ OSN to identify
    int K=7;
    vector<bool> osn_vbInliers;
    OSN osn(kp1, img1size, kp2, img2size, corres, MethodPara(2.2, K, 2.0));
	int num_inliers = osn.GetInlierMask(osn_vbInliers);
//     for(int i=0;i<kp1.size();i++){
//         cout<<kp1[kp1.size()-1]<<endl;
//     }
}