// This is an advanced implementation of the algorithm described in the following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014. 

// Modifier: Tong Qin               qintonguav@gmail.com
// 	         Shaozu Cao 		    saozu.cao@connect.ust.hk


// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/*
    .....点云特征提取......
    
    订阅激光雷达点云话题
    
    点云预期处理（Nan inf去除）、雷达点云整理（整理到每线）、曲率排序、发布5个点云话题
*/

#include <cmath>
#include <vector>
#include <string>
#include "aloam_velodyne/common.h"
#include "aloam_velodyne/tic_toc.h"
#include <nav_msgs/Odometry.h>
// #include <opencv/cv.h>     //本机用的ubuntu20.04----opencv4  
#include <opencv2/imgproc.hpp> //因此需要调整cv头文件
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

using std::atan2;
using std::cos;
using std::sin;

const double scanPeriod = 0.1;//扫描周期, velodyne频率10Hz，周期0.1s
const int systemDelay = 0;//弃用前systemDelay帧初始数据
int systemInitCount = 0;//systemInitCount用于计数过了多少帧
bool systemInited = false;//超过systemDelay后，systemInited为true即初始化完成
int N_SCANS = 0;//激光雷达线数初始化为0
float cloudCurvature[400000];//点云曲率, 400000为一帧点云中点的最大数量
//如果将velodyne16雷达的扫描频率设置为10Hz，那么一秒就返回10帧数据。
//工作在10Hz的频率下，这个雷达的水平扫描角度的分辨率是0.2°，我们可以算出来理论上一帧有 360/0.2 × 16 = 28800
//但是实际上每次的点数不是完全一样的，在程序中将存储点的数组定义为40000个元素。
int cloudSortInd[400000];//曲率点对应的序号
int cloudNeighborPicked[400000];//点是否筛选过标志：0-未筛选过，1-筛选过
//点分类标号:2-代表曲率很大，1-代表曲率比较大,-1-代表曲率很小，0-曲率比较小(其中1包含了2,0包含了1,0和1构成了点云全部的点)
int cloudLabel[400000];

//两点曲率比较
bool comp (int i,int j) { return (cloudCurvature[i]<cloudCurvature[j]); }
//设置发布内容
ros::Publisher pubLaserCloud;//整体点云
ros::Publisher pubCornerPointsSharp;//角点
ros::Publisher pubCornerPointsLessSharp;//降采样角点
ros::Publisher pubSurfPointsFlat;//面点
ros::Publisher pubSurfPointsLessFlat;//降采样面点
ros::Publisher pubRemovePoints;//剔除点
//ros形式的一线扫描
std::vector<ros::Publisher> pubEachScan;

bool PUB_EACH_LINE = false;//是否发布每行Scan

double MINIMUM_RANGE = 0.1; //根据距离去除过远的点，距离的参数

/*
    removeClosedPointCloud函数
    去除近距离点，衡量标准<thres
*/
template <typename PointT>//过近点去除 使用template进行兼容,兼容不同类型的输入
void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                              pcl::PointCloud<PointT> &cloud_out, float thres)
{
    //统一header(时间戳)和size
    if (&cloud_in != &cloud_out)
    {
        cloud_out.header = cloud_in.header;
        cloud_out.points.resize(cloud_in.points.size());
    }

    size_t j = 0;
    //逐点距离比较  剔除近距离点（<thres）
    for (size_t i = 0; i < cloud_in.points.size(); ++i)
    {
        if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
            continue;
        cloud_out.points[j] = cloud_in.points[i];
        j++;
    }
    if (j != cloud_in.points.size())
    {
        cloud_out.points.resize(j);
    }

    cloud_out.height = 1;    //数据行数，默认1为无组织的数据
    cloud_out.width = static_cast<uint32_t>(j);//点云个数标识width*height
    cloud_out.is_dense = true; //如果为false，则表示点云中包含inf/NaN这样的点
}

/*
    激光雷达回调函数
    特征配准，并输出5个类型的点云
*/
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
{
    if (!systemInited)//系统初始化，延时启动
    { 
        systemInitCount++;
        if (systemInitCount >= systemDelay)
        {
            systemInited = true;
        }
        else
            return;
    }

    //定义计时
    TicToc t_whole;//计算整个回调函数的时间
    TicToc t_prepare;//计算雷达点云有序化的时间
    std::vector<int> scanStartInd(N_SCANS, 0);//记录每个scan有曲率的点的开始
    std::vector<int> scanEndInd(N_SCANS, 0);//和结束索引

    pcl::PointCloud<pcl::PointXYZ> laserCloudIn;//命名一个pcl形式的输入点云
    pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);//把输入的ros形式点云转化为pcl形式
    std::vector<int> indices;

    // 对点云滤波，去除NaN值得无效点云， 去除 MINIMUM_RANGE 距离以内的点
    pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
    removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

    /*
    150-269行  --由于velodyne激光雷达扫描方式问题，将点云规划到每条线上。
    */
    int cloudSize = laserCloudIn.points.size();
    //点云起始点和终止点（用atan2因为【-pi pi】,而atan是【-pi/2 pi/2】,负号是因为velodyne是顺时针旋转）
    float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
    float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y,
                          laserCloudIn.points[cloudSize - 1].x) +
                   2 * M_PI;
    //取0-2pi  -2pi-0  防止角度异常
    if (endOri - startOri > 3 * M_PI)
    {
        endOri -= 2 * M_PI;
    }
    else if (endOri - startOri < M_PI)
    {
        endOri += 2 * M_PI;
    }
    //printf("end Ori %f\n", endOri);

    bool halfPassed = false;//过半flag
    int count = cloudSize;//点数
    PointType point;    //按线数保存的点云集合
    std::vector<pcl::PointCloud<PointType>> laserCloudScans(N_SCANS);
    for (int i = 0; i < cloudSize; i++)
    {
        point.x = laserCloudIn.points[i].x;
        point.y = laserCloudIn.points[i].y;
        point.z = laserCloudIn.points[i].z;
	    //通过计算垂直视场角确定激光点在哪个扫描线上（N_SCANS线激光雷达）
        float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
        int scanID = 0;
	    //根据不同线数使用不同参数对每个点对应的第几根激光线进行判断
        if (N_SCANS == 16)
        {
            scanID = int((angle + 15) / 2 + 0.5);
            if (scanID > (N_SCANS - 1) || scanID < 0) //去除错误点
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 32)
        {
            scanID = int((angle + 92.0/3.0) * 3.0 / 4.0);
            if (scanID > (N_SCANS - 1) || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else if (N_SCANS == 64)
        {   
            if (angle >= -8.83)
                scanID = int((2 - angle) * 3.0 + 0.5);
            else
                scanID = N_SCANS / 2 + int((-8.83 - angle) * 2.0 + 0.5);

            // use [0 50]  > 50 remove outlies 
            if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
            {
                count--;
                continue;
            }
        }
        else
        {
            printf("wrong scan number\n");
            ROS_BREAK();
        }
        //printf("angle %f scanID %d \n", angle, scanID);
	//根据扫描线是否旋转过半选择与起始位置还是终止位置进行差值计算，从而进行补偿
        float ori = -atan2(point.y, point.x);
        if (!halfPassed)
        {             //确保-pi/2 < ori - startOri < 3*pi/2
            if (ori < startOri - M_PI / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > startOri + M_PI * 3 / 2)
            {
                ori -= 2 * M_PI;
            }

            if (ori - startOri > M_PI)
            {
                halfPassed = true;
            }
        }
        else
        {        //确保-3*pi/2 < ori - endOri < pi/2
            ori += 2 * M_PI;
            if (ori < endOri - M_PI * 3 / 2)
            {
                ori += 2 * M_PI;
            }
            else if (ori > endOri + M_PI / 2)
            {
                ori -= 2 * M_PI;
            }
        }
        //看看旋转多少了，记录比例relTime
        float relTime = (ori - startOri) / (endOri - startOri);
        //第几根线和本线进度到多少记录在point.intensity
        point.intensity = scanID + scanPeriod * relTime;
        //按线分类保存
        laserCloudScans[scanID].push_back(point); 
    }
    
    cloudSize = count;//打印有效点
    printf("points size %d \n", cloudSize);
   // 全部集合到一个点云里面去，但是使用两个数组标记起始和结果
    //这里分别+5和-6是为了计算曲率方便
    pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
    for (int i = 0; i < N_SCANS; i++)
    { 
        scanStartInd[i] = laserCloud->size() + 5;
        *laserCloud += laserCloudScans[i];
        scanEndInd[i] = laserCloud->size() - 6;
    }
// 将一帧无序点云转换成有序点云消耗的时间，这里指的是前面处理雷达数据的时间
    printf("prepare time %f \n", t_prepare.toc());

    for (int i = 5; i < cloudSize - 5; i++)
    { 
        //计算每个点的曲率
        float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
        float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
        float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
        cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;
        cloudSortInd[i] = i;//默认曲率排序
        cloudNeighborPicked[i] = 0;
        cloudLabel[i] = 0;//曲率标签
    }

/*
曲率计算完成后进行特征分类，提取特征点有几点原则：
    1.为了提高效率，每条扫描线分成6个扇区，在每个扇区内，选取极大边线点（2个）和次极大边线点（20个）
    2. 寻找曲率最小的4个点，作为极小平面点，剩下未被标记的点，全部作为次极小平面点。
    3. 对于次极小平面点，由于点数太多，采取体素栅格滤波降采样的方法避免过多聚堆。
*/
    TicToc t_pts;//定义计时点（特征提取时间）

    pcl::PointCloud<PointType> cornerPointsSharp;//角点（曲率max）
    pcl::PointCloud<PointType> cornerPointsLessSharp;//次角点（曲率less max）
    pcl::PointCloud<PointType> surfPointsFlat;//面点（曲率min）
    pcl::PointCloud<PointType> surfPointsLessFlat;//次面点（less min）

    float t_q_sort = 0;
    for (int i = 0; i < N_SCANS; i++)
    {// 如果最后一个可算曲率的点与第一个的数量差小于6，说明无法分成6个扇区，跳过
        if( scanEndInd[i] - scanStartInd[i] < 6)
            continue;
        // 用来存储次极小平面点，后面会进行降采样
        pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
        for (int j = 0; j < 6; j++)
        {
            //6等分一条scan线
            int sp = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * j / 6; 
            int ep = scanStartInd[i] + (scanEndInd[i] - scanStartInd[i]) * (j + 1) / 6 - 1;

            TicToc t_tmp;
            std::sort (cloudSortInd + sp, cloudSortInd + ep + 1, comp);//升序排序
            t_q_sort += t_tmp.toc();// t_q_sort累计每个扇区曲率排序时间总和

            int largestPickedNum = 0; // 选取极大边线点（2个）和次极大边线点（20个）
            for (int k = ep; k >= sp; k--)
            {
                //由于排序了，原cloudSortInd序号变换，，用ind代替做指引
                int ind = cloudSortInd[k]; 
                //未被选 && 曲率>0.1
                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] > 0.1)
                {
                    largestPickedNum++; //最大选择点+1
                    if (largestPickedNum <= 2)//选最大2个
                    {                        
                        cloudLabel[ind] = 2;//最大点标签为2（代表角点）
                        cornerPointsSharp.push_back(laserCloud->points[ind]);//放入max
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);//同时放入less max
                    }
                    else if (largestPickedNum <= 20)//选次大20个
                    {                        
                        cloudLabel[ind] = 1; 
                        cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                    }
                    else//20个点选完结束
                    {
                        break;
                    }

                    cloudNeighborPicked[ind] = 1; //已选择
                    // 为了保证特征点不过度集中，将选中的点周围5个点都置1,避免后续会选到
                    for (int l = 1; l <= 5; l++)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            //如果与特征点距离>0.05 就全部跳过
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            int smallestPickedNum = 0;// 选取min面点，同上循环一样
            for (int k = sp; k <= ep; k++)
            {
                int ind = cloudSortInd[k];

                if (cloudNeighborPicked[ind] == 0 &&
                    cloudCurvature[ind] < 0.1)
                {
                    cloudLabel[ind] = -1; //标签置为-1
                    surfPointsFlat.push_back(laserCloud->points[ind]);

                    smallestPickedNum++;
                    if (smallestPickedNum >= 4)//只有选4个
                    { 
                        break;
                    }

                    cloudNeighborPicked[ind] = 1;
                    for (int l = 1; l <= 5; l++)
                    { 
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                    for (int l = -1; l >= -5; l--)
                    {
                        float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                        float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                        float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                        if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                        {
                            break;
                        }

                        cloudNeighborPicked[ind + l] = 1;
                    }
                }
            }

            for (int k = sp; k <= ep; k++)
            {
                if (cloudLabel[k] <= 0)
                {
                    //将<=0全部归入平面点中less flat类别中
                    surfPointsLessFlatScan->push_back(laserCloud->points[k]);
                }
            }
        }
        //由于less flat点最多，对每个分段less flat的点进行体素栅格滤波
        //主要作用是对点云进行降采样，可以在保证点云原有几何结构基本不变的前提下减少点的数量
        pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
        pcl::VoxelGrid<PointType> downSizeFilter;
        downSizeFilter.setInputCloud(surfPointsLessFlatScan);
        downSizeFilter.setLeafSize(0.2, 0.2, 0.2);
        downSizeFilter.filter(surfPointsLessFlatScanDS);

        surfPointsLessFlat += surfPointsLessFlatScanDS;
    }
    printf("sort q time %f \n", t_q_sort);//排序时间
    printf("seperate points time %f \n", t_pts.toc());//特征提取时间

    //点云输出
    sensor_msgs::PointCloud2 laserCloudOutMsg;// 创建publish msg实例
    pcl::toROSMsg(*laserCloud, laserCloudOutMsg);// 有序点云转化为msg
    laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;// 时间戳保持不变
    laserCloudOutMsg.header.frame_id = "/camera_init"; // frame_id名字，坐标系
    pubLaserCloud.publish(laserCloudOutMsg);

    sensor_msgs::PointCloud2 cornerPointsSharpMsg;
    pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
    cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsSharp.publish(cornerPointsSharpMsg);

    sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
    pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
    cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
    cornerPointsLessSharpMsg.header.frame_id = "camera_init";
    pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);

    sensor_msgs::PointCloud2 surfPointsFlat2;
    pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
    surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsFlat2.header.frame_id = "camera_init";
    pubSurfPointsFlat.publish(surfPointsFlat2);

    sensor_msgs::PointCloud2 surfPointsLessFlat2;
    pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
    surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
    surfPointsLessFlat2.header.frame_id = "camera_init";
    pubSurfPointsLessFlat.publish(surfPointsLessFlat2);

    // 可以按照每个scan发出去，不过这里是false
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i< N_SCANS; i++)
        {
            sensor_msgs::PointCloud2 scanMsg;
            pcl::toROSMsg(laserCloudScans[i], scanMsg);
            scanMsg.header.stamp = laserCloudMsg->header.stamp;
            scanMsg.header.frame_id = "camera_init";
            pubEachScan[i].publish(scanMsg);
        }
    }

    //总时间输出
    printf("scan registration time %f ms *************\n", t_whole.toc());
    if(t_whole.toc() > 100)
        ROS_WARN("scan registration process over 100ms");
}

/*
    主函数
    订阅雷达话题
    注册发布点云话题
*/
int main(int argc, char **argv)
{
    ros::init(argc, argv, "scanRegistration");//节点名称
    ros::NodeHandle nh;//注册ROS句柄

    //从launch文件参数服务器中获取多少线的激光雷达，如果没有则默认
    nh.param<int>("scan_line", N_SCANS, 16);//默认16线
    nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);//最小扫描距离MINIMUM_RANGE，默认0.1
    printf("scan line number %d \n", N_SCANS);
    // 算法只支持16/32/64线激光雷达
    if(N_SCANS != 16 && N_SCANS != 32 && N_SCANS != 64)
    {
        printf("only support velodyne with 16, 32 or 64 scan line!");
        return 0;
    }
    // 订阅初始的激光雷达数据，并注册回调函数laserCloudHandler
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/ns1/velodyne_points", 100, laserCloudHandler);
    // 发布话题：有序点云（删除过近点、设置索引）
    pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2", 100);
    //曲率max点集合(角点)
    pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 100);
    //次极大边线点集合
    pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 100);
    //曲率min点集合平面点)
    pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 100);
    //次极小平面点集合
    pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 100);
    //删除的点云
    pubRemovePoints = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 100);
    //PUB_EACH_LINE = false
    if(PUB_EACH_LINE)
    {
        for(int i = 0; i < N_SCANS; i++)
        {
            ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
            pubEachScan.push_back(tmp);
        }
    }
    ros::spin();// 循环执行回调函数

    return 0;
}
