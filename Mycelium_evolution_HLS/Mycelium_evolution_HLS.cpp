#include "TwoD_evolution_design_chatzip.h"
#include <hls_vector.h>
//#include <hls_stream.h>
//#include <iostream>
//#include <fstream>
//#include <fstream>
//#include <sstream>
//#include <string>
//#include <stdint.h>

void evolution(datat u[100][100], datat v[100][100], datat c[100][100],datat u_new[100][100], datat v_new[100][100], datat c_new[100][100], datat ij_mat[100][100],datat n[100][100],bool code_selector){

//#pragma HLS ARRAY_PARTITION variable=u type=complete dim=2
//#pragma HLS ARRAY_PARTITION variable=v type=complete dim=2
//#pragma HLS ARRAY_PARTITION variable=c type=complete dim=2

//#pragma HLS bind_storage variable=ij_mat type=RAM_T2P
//#pragma HLS bind_storage variable=u type=RAM_T2P
//#pragma HLS bind_storage variable=v type=RAM_T2P
//#pragma HLS bind_storage variable=c type=RAM_T2P
//#pragma HLS ARRAY_PARTITION variable=v_new type=block factor=4

			const int gridSize = 100;
			/*Reaction-Diffusion parameters*/

			datat pa = 0.5;
			datat pb = 0.8;
			datat pc = 0.16;
			datat pe = 2.6;
			datat d = 30;
			datat dt = 0.1;
			datat threshold = 0.5;
			datat pk = 0.05;
			datat ph = 1;
			//datat alpha[gridSize][gridSize];
			datat alpha = 1;
			datat amax = 20;
			datat smax = 30;
			datat width = 10;
			datat kati = 0.7;
			datat_gamma gamma = 625;
			//int i = 0;
			//int j = 0;
	//		int w = 0;
	//		int p = 0;
			int k = 5;

			int mid = int(gridSize/2);
			const int totalIterations=120;

			//static datat n[gridSize][gridSize];
			//static datat ij_mat[100][100];
			//static datat u_new[gridSize][gridSize];
			//static datat v_new[gridSize][gridSize];
			//static datat c_new[gridSize][gridSize];
			static datat f_uv[gridSize][gridSize];
			static datat g_uv[gridSize][gridSize];
//#pragma HLS BIND_STORAGE variable=f_uv type=RAM_T2P
//#pragma HLS BIND_STORAGE variable=g_uv type=RAM_T2P

//#pragma HLS ARRAY_PARTITION variable=ij_mat type=complete dim=0
//#pragma HLS bind_storage variable=ij_mat type=RAM_T2P impl=uram

//#pragma HLS DONT_TOUCH variable=f_uv

/*			if(steps==0){
				for (i=0;i<gridSize;i++)
				{
					for (j=0;j<gridSize;j++)
					{
					//	u[i][j] = datat(0);
					//	v[i][j] = datat(0);
					//	c[i][j] = datat(0);
					//	ij_mat[i][j] = datat(0);
					}
				}

*/
	//		}else{

			datat L[3][3];
			datat lap_side = 0.35;
			datat lap_diag = 0.1;
			datat lap = 1.0/9;

			datat L_normal[3][3] =
					{
					lap_diag, lap_side, lap_diag,
					lap_side, -lap, lap_side,
					lap_diag, lap_side, lap_diag
					};

			datat L_tips[3][3] =
						{
						datat(0), datat(0.05), datat(0),
						datat(0.05), datat(-0.2), datat(0.05),
						datat(0), datat(0.05), datat(0)
						};

			datat ij_mat_kernel_tips[9][9] =
			{
				0, 0, 0, 0, 1, 0, 0, 0, 0,
				0, 0, 0, 1, 1, 1, 0, 0, 0,
				0, 0, 1, 1, 1, 1, 1, 0, 0,
				0, 0, 1, 1, 1, 1, 1, 0, 0,
				1, 0, 1, 1, 1, 1, 1, 0, 1,
				0, 0, 1, 1, 1, 1, 1, 0, 0,
				0, 0, 1, 1, 1, 1, 1, 0, 0,
				0, 0, 0, 1, 1, 1, 0, 0, 0,
				0, 0, 0, 0, 1, 0, 0, 0, 0
					};


			datat ij_mat_kernel_normal[5][5] =
						{lap_diag,   lap_side,   lap_side,   lap_side,   lap_diag,
					     lap_side,   lap_diag,   lap_side,   lap_diag,   lap_side,
					     lap_side,   lap_side,   lap,        lap_side,   lap_side,
					     lap_side,   lap_diag,   lap_side,   lap_diag,   lap_side,
					     lap_diag,   lap_side,   lap_side,   lap_side,   lap_diag
						};

//			for (i=0;i<gridSize;i++)
//				{
//					for (j=0;j<gridSize;j++)
//					{
//						//ij_mat[i][j] = datat(0);
//						//n[i][j] = datat(1);
//						//u_new[i][j] = datat(0);
//						//v_new[i][j] = datat(0);
//						//c_new[i][j] = datat(0);
//						//f_uv[i][j] = datat(0);
//						//g_uv[i][j] = datat(0);
//					}
//				}
//		for (i=0;i<gridSize;i++)
//				{
//				for (j=mid-1;j<gridSize;j++)
//					{
//						n[i][j] = datat(0.3);
//					}
//				}
			if (code_selector == true){
				width = datat(1);
				for (int w1=0;w1<3;w1++)
					{
#pragma HLS unroll
					for (int p1=0;p1<3;p1++)
					{
						L[w1][p1] = L_normal[w1][p1];
					}
				}
			}
			else if (code_selector == false){
				width = datat(10);
				for (int w2=0;w2<3;w2++)
					{
#pragma HLS unroll
					for (int p2=0;p2<3;p2++)
					{
						L[w2][p2] = L_tips[w2][p2];
					}
				}
			}

			for(int i=1;i<gridSize-1;i++)
			{
				for(int j=1;j<gridSize-1;j++)
				{
					if (code_selector == true)
					{
						if (i>=2 && j>=2 && i<(gridSize-2) && j<(gridSize-2))
						{
						if(c[i][j] > 0)
						{
											ij_mat[i][j] = lap;
												 	 ij_mat[i-1][j-1] = lap_diag;
													 ij_mat[i-1][j] = lap_side;
													 ij_mat[i-1][j+1] = lap_diag;
													 ij_mat[i][j-1] = lap_side;
													 ij_mat[i][j+1] = lap_side;
													 ij_mat[i+1][j-1] = lap_diag;
													 ij_mat[i+1][j+1] = lap_diag;
													 ij_mat[i+1][j] = lap_side;

													 ij_mat[i-2][j-2] = lap_diag;
													 ij_mat[i-2][j] = lap_side;
													 ij_mat[i-2][j+2] = lap_diag;
													 ij_mat[i][j-2] = lap_side;
													 ij_mat[i][j+2] = lap_side;
													 ij_mat[i+2][j-2] = lap_diag;
													 ij_mat[i+2][j+2] = lap_diag;
													 ij_mat[i+2][j] = lap_side;

													 ij_mat[i-2][j-1] = lap_side;
													 ij_mat[i-2][j+1] = lap_side;
													 ij_mat[i+2][j-1] = lap_side;
													 ij_mat[i+2][j+1] = lap_side;

													 ij_mat[i-1][j-2] = lap_side;
													 ij_mat[i-1][j+2] = lap_side;
													 ij_mat[i+1][j-2] = lap_side;
													 ij_mat[i+1][j+2] = lap_side;
//							for (int w3=-2;w3<=2;w3++)
//							{
////#pragma HLS PERFORMANCE target_tl=6
////#pragma HLS pipeline II=5 rewind style=stp
//								for (int p3=-2;p3<=2;p3++)
//								{
//									if(ij_mat_kernel_normal[w3+2][p3+2]>0)
//									{
//										ij_mat[i+w3][j+p3] = ij_mat_kernel_normal[w3+2][p3+2];
//									}
//								}
//							}
						}
					}
					}
					else if (code_selector == false) // ME TIPS
					{
						if (i>=4 && j>=4 && i<(gridSize-4) && j<(gridSize-4))
						{
						if(c[i][j] > 0)
						{
							 	 	 	 	 	 	 ij_mat[i][j] = lap;
													 ij_mat[i-1][j-1] = lap_side;
													 ij_mat[i-1][j] = lap_side;
													 ij_mat[i-1][j+1] = lap_side;
													 ij_mat[i][j-1] = lap_side;
													 ij_mat[i][j+1] = lap_side;
													 ij_mat[i+1][j-1] = lap_side;
													 ij_mat[i+1][j+1] = lap_side;
													 ij_mat[i+1][j] = lap_side;
													 ij_mat[i-2][j-2] = lap_side;
													 ij_mat[i-2][j] = lap_side;
													 ij_mat[i-2][j+2] = lap_side;
													 ij_mat[i][j-2] = lap_side;
													 ij_mat[i][j+2] = lap_side;
													 ij_mat[i+2][j-2] = lap_side;
													 ij_mat[i+2][j+2] = lap_side;
													 ij_mat[i+2][j] = lap_side;
													 ij_mat[i-3][j-1] = lap_side;
													 ij_mat[i-3][j] = lap_side;
													 ij_mat[i-3][j+1] = lap_side;
													 ij_mat[i+3][j-1] = lap_side;
													 ij_mat[i+3][j] = lap_side;
													 ij_mat[i+3][j+1] = lap_side;
													 ij_mat[i-4][j] = lap_side;
													 ij_mat[i+4][j] = lap_side;
													 ij_mat[i][j-4] = lap_side;
													 ij_mat[i][j+4] = lap_side;
													 ij_mat[i-2][j-1] = lap_side;
													 ij_mat[i-2][j+1] = lap_side;
													 ij_mat[i+2][j-1] = lap_side;
													 ij_mat[i+2][j+1] = lap_side;
													 ij_mat[i-3][j] = lap_side;
													 ij_mat[i+3][j] = lap_side;
													 ij_mat[i-1][j-2] = lap_side;
													 ij_mat[i-1][j+2] = lap_side;
													 ij_mat[i+1][j-2] = lap_side;
													 ij_mat[i+1][j+2] = lap_side;
//							for (int w4=-4;w4<=4;w4++)
//								{
////#pragma HLS PERFORMANCE target_tl=6
////#pragma HLS pipeline II=5 rewind style=stp
//								for (int p4=-4;p4<=4;p4++)
//									{
//									if(ij_mat_kernel_tips[w4+4][p4+4]>0)
//									{
//										ij_mat[i+w4][j+p4] = ij_mat_kernel_tips[w4+4][p4+4];
//									}
//								}
//							}
						}
					}
					}
						f_uv[i][j] = width*(pa*u[i][j]+u[i][j]*u[i][j]-pb*u[i][j]*v[i][j])*n[i][j];
						g_uv[i][j] = pe*u[i][j]*u[i][j]*u[i][j] - v[i][j];
						v_new[i][j] = v[i][j] + dt * ij_mat[i][j]*(d * (L[2][1] * v[i+1][j] + L[0][1] * v[i-1][j] + L[1][2] * v[i][j+1] + L[1][0] * v[i][j-1] + L[0][0] *v[i-1][j-1] + L[2][0] *v[i+1][j-1] + L[0][2] *v[i-1][j+1] + L[2][2] *v[i+1][j+1] + L[1][1] * v[i][j]) + gamma * g_uv[i][j]);
						u_new[i][j] = u[i][j] + dt * ij_mat[i][j]*((L[2][1] * u[i+1][j] + L[0][1] * u[i-1][j] + L[1][2] * u[i][j+1] + L[1][0] * u[i][j-1] + L[0][0] *u[i-1][j-1] + L[2][0] *u[i+1][j-1] + L[0][2] *u[i-1][j+1] + L[2][2] *u[i+1][j+1] + L[1][1] * u[i][j]) + gamma * f_uv[i][j]);
						c_new[i][j] = c[i][j] + dt * ij_mat[i][j] * (gamma*ph*c[i][j]*(alpha-c[i][j])*(c[i][j] - kati));

						if(u[i][j] <= threshold)
						{
							//	v_new[i][j] = v[i][j] + dt * ij_mat[i][j]*(d * (L[2][1] * v[i+1][j] + L[0][1] * v[i-1][j] + L[1][2] * v[i][j+1] + L[1][0] * v[i][j-1] + L[0][0] *v[i-1][j-1] + L[2][0] *v[i+1][j-1] + L[0][2] *v[i-1][j+1] + L[2][2] *v[i+1][j+1] + L[1][1] * v[i][j]) + gamma * g_uv[i][j]);
							alpha = datat(0.49);
						}
						else
						{
							alpha = datat(0.49) - datat(3) * (u[i][j] - threshold);
							v_new[i][j] = 0;
						}
						if (v_new[i][j]>smax){
							v_new[i][j] = smax;
						}//else{
						//v_new[i][j] = v[i][j] + dt * ij_mat[i][j]*(d * (L[2][1] * v[i+1][j] + L[0][1] * v[i-1][j] + L[1][2] * v[i][j+1] + L[1][0] * v[i][j-1] + L[0][0] *v[i-1][j-1] + L[2][0] *v[i+1][j-1] + L[0][2] *v[i-1][j+1] + L[2][2] *v[i+1][j+1] + L[1][1] * v[i][j]) + gamma * g_uv[i][j]);
							//}


						if (u_new[i][j]<0){
							u_new[i][j] = datat(0);
							}//else if(u_new[i][j]>amax){
												//	u_new[i][j] = amax;
												//}else{
												//	u_new[i][j] = u[i][j] + dt * ij_mat[i][j]*((L[2][1] * u[i+1][j] + L[0][1] * u[i-1][j] + L[1][2] * u[i][j+1] + L[1][0] * u[i][j-1] + L[0][0] *u[i-1][j-1] + L[2][0] *u[i+1][j-1] + L[0][2] *u[i-1][j+1] + L[2][2] *u[i+1][j+1] + L[1][1] * u[i][j]) + gamma * f_uv[i][j]);
												//}


						if (u_new[i][j]>amax){
						u_new[i][j] = amax;
						}

						if (alpha < 0){
							c_new[i][j] = datat(1);
						}
					//	v[i][j] = v_new[i][j];
					//	u[i][j] = u_new[i][j];
					//	c[i][j] = c_new[i][j];

					//std::cout << c[i][j] << " ";
				}
				//std::cout << std::endl;
			}
			//std::cout << steps << std::endl;

		//return c[1][1];
	//}
}


datat evolution_top(bool load){

const int gridSize=100;
//static datat u[gridSize][gridSize];
//static datat v[gridSize][gridSize];
//static datat c[gridSize][gridSize];

//#pragma HLS array_partition variable=u type=complete dim=0
//#pragma HLS array_partition variable=v type=complete dim=0
//#pragma HLS array_partition variable=c type=complete dim=0

static datat u[100][100];
static datat v[100][100];
static datat c[100][100];
datat u_new[100][100];
datat v_new[100][100];
datat c_new[100][100];
datat ij_mat[100][100];
datat n[100][100];
int mid=50;
int k=5;
int steps;

const int totalIterations=120;
int narrow_iterations_limit1 = datat(1.0/4)*totalIterations;
	int narrow_iterations_limit2_tips = datat(1.3/4)*totalIterations;
	int narrow_iterations_limit3 = datat(2.3/4)*totalIterations;
	int narrow_iterations_limit4_tips = datat(2.6/4)*totalIterations;
	int narrow_iterations_limit5 = datat(3.6/4)*totalIterations;

	hls::vector<bool, totalIterations> code_selector;


if(load==1){

steps=0;
	//hls::vector<bool, totalIterations> code_selector = false;

	for (int i = 0; i < narrow_iterations_limit1; i++)
	    code_selector[i] = true;
	for (int i = narrow_iterations_limit2_tips; i <= narrow_iterations_limit3; i++)
	    code_selector[i] = true;
	for (int i = narrow_iterations_limit4_tips; i <= narrow_iterations_limit5; i++)
	    code_selector[i] = true;
	//std::string filename="data_FIN_" + std::to_string(steps)+".csv";
	//	std::ofstream myFile(filename);

					for(int i=0;i<100;i++)
			    		{
			    			for (int j=0;j<100;j++)
			    			{
			    				u[i][j]=0;
			    				v[i][j]=0;
			    				c[i][j]=0;
			    				u_new[i][j]=0;
			    				v_new[i][j]=0;
			    				c_new[i][j]=0;
			    				ij_mat[i][j] = datat(0);
			    				n[i][j]=1;
			    		//		std::cout << c[i][j] << " ";
			    		//					myFile << c[i][j] << ",";
			    			}
			    		//	myFile << std::endl;
			    		}
					//myFile.close();
					for (int i=0;i<gridSize;i++)
						{
						for (int j=mid-1;j<gridSize;j++)
							{
								n[i][j] = datat(0.3);
							}
						}

						for (int i=mid-k-1;i<mid+k;i++)
						{
							for (int j=mid-k-1;j<mid+k;j++)
							{
								u[i][j]=datat(0.5+0.05);
								v[i][j]=datat(0.1+0.05);
								c[i][j]=datat(1);
							}
						}
}
else{
	evolution(u,v,c,u_new,v_new,c_new,ij_mat,n,code_selector[steps]);
	steps=steps+1;
//	std::string filename="data_FIN_" + std::to_string(steps)+".csv";
//	std::ofstream myFile(filename);

	for(int i=0;i<100;i++)
	{
		for(int j=0;j<100;j++)
		{

			u[i][j]=u_new[i][j];
			v[i][j]=v_new[i][j];
			c[i][j]=c_new[i][j];
//			std::cout << c[i][j] << " ";
//			myFile << c[i][j] << ",";
//			std::cout << c[i][j];
		}
//		std::cout << std::endl;
//		myFile << std::endl;
	}
//	std::cout << std::endl << steps;
//	myFile.close();
}
//std::cout <<c[60][35]<< std::endl;
return c[60][35];
}
