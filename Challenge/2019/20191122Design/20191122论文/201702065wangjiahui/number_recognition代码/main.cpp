#define _CRT_SECURE_NO_WARNINGS
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

class myRect
{
public:
	myRect(){}
	~myRect(){}
	myRect(Rect &temp):myRc(temp){}
	//�ȽϾ������Ͻǵĺ����꣬�Ա�����
	bool operator<(myRect &rect)
	{
		if (this->myRc.x < rect.myRc.x)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	//���ظ�ֵ�����
	myRect operator=(myRect &rect)
	{
		this->myRc = rect.myRc;
		return *this;
	}
	//��ȡ����
	Rect getRect()
	{
		return myRc;
	}
private:
	Rect myRc;//��ž���
};

//��ͼƬ�����غ�
int getPiexSum(Mat &image)
{
	int sum = 0;
	for (int i = 0; i < image.cols; i++)
	{
		for (int j = 0; j < image.rows; j++)
		{
			sum += image.at<uchar>(j, i);
		}
	}
	return sum;
}

/*������*/
int main()
{
	//����Ҫʶ���ͼƬ������ʾ
	Mat srcImage = imread("number3.jpg");
	imshow("ԭͼ", srcImage);
	//��ͼ����д���ת��Ϊ�Ҷ�ͼȻ����תΪ��ֵͼ
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat binImage;
	//��4������ΪCV_THRESH_BINARY_INV����Ϊ�ҵ�����ԭͼΪ�׵׺���
	//��Ϊ�ڵװ�����ѡ��CV_THRESH_BINARY����
	threshold(grayImage, binImage, 100, 255, CV_THRESH_BINARY_INV);

	//Ѱ������������ָ��ΪѰ���ⲿ��������Ȼһ�����ֿ����ж��������ɣ�����4,6,8,9������
	Mat conImage = Mat::zeros(binImage.size(), binImage.type());
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	//ָ��CV_RETR_EXTERNALѰ�����ֵ�������
	findContours(binImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//��������
	drawContours(conImage, contours, -1, 255);

	//��ÿ�����֣����뿪�����浽������
	vector<myRect> sort_rect;
	for (int i = 0; i < contours.size(); i++)
	{
		//boundingRect������������Ӿ���
		Rect tempRect = boundingRect(contours[i]);
		sort_rect.push_back(tempRect);
	}

	//�Ծ��ν���������Ϊ������˳��һ��������������˳��
	for (int  i = 0; i < sort_rect.size(); i++)
	{
		for (int j = i + 1; j < sort_rect.size(); j++)
		{
			if (sort_rect[j] < sort_rect[i])
			{
				myRect temp = sort_rect[j];
				sort_rect[j] = sort_rect[i];
				sort_rect[i] = temp;
			}
		}
	}


	/*����ģ�壬��û�������Լ��½�һ��*/

	//�½�,����һ�ξͺã���������ģ��Ĳ���Ϊ0-9ʮ�����ֵ�ͼ��
	//for (int i = 0; i < 10; i++)
	//{
	//	Mat ROI = conImage(sort_rect[i].getRect());
	//	Mat dstROI;
	//	resize(ROI, dstROI, Size(40, 50),0, 0, INTER_NEAREST);
	//	char name[64];
	//	sprintf(name, "C:/Users/Administrator/Desktop/number_recognition/number_recognition/image/%d.jpg", i);
	//	//imshow(str, dstROI);
	//	imwrite(name, dstROI);
	//}

	//����ģ��
	vector<Mat> myTemplate;
	for (int i = 0; i < 10; i++)
	{
		char name[64];
		sprintf(name, "number2.jpg", i);
		Mat temp = imread(name, 0);
		myTemplate.push_back(temp);
	}

	//��˳��ȡ���ͷָ�����
	vector<Mat> myROI;
	for (int i = 0; i < sort_rect.size(); i++)
	{
		Mat ROI;
		ROI = conImage(sort_rect[i].getRect());
		Mat dstROI = Mat::zeros(myTemplate[0].size(),myTemplate[0].type());
		resize(ROI, dstROI, myTemplate[0].size(), 0, 0, INTER_NEAREST);
		myROI.push_back(dstROI);
	}

	//���бȽ�,��ͼƬ��ģ�������Ȼ����ȫ�����غͣ�����С��ʾԽ���ƣ��������ƥ��
	vector<int> seq;//˳����ʶ����
	for (int i = 0; i < myROI.size(); i++)
	{
		Mat subImage;
		int sum = 0;
		int min = 100000;
		int min_seq = 0;//��¼��С�ĺͶ�Ӧ������
		for (int j = 0; j < 10; j++)
		{
			//��������ͼƬ�Ĳ�ֵ
			absdiff(myROI[i], myTemplate[j], subImage);
			sum = getPiexSum(subImage);
			if (sum < min)
			{
				min = sum;
				min_seq = j;
			}
			sum = 0;
		}
		seq.push_back(min_seq);
	}

	//������
	cout << "ʶ����Ϊ��";
	for (int i = 0; i < seq.size(); i++)
	{
		cout << seq[i];
	}
	cout << endl;

	waitKey(0);
	system("pause");
	return 0;
}