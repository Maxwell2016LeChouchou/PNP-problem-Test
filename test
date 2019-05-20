/*

	Description: Given the unitary vectors from camera to the 3 feature points, compute the absolute pose of a camera using 
		three 3D-to-2D correspondences, as a compensative part for our PnP project

	Reference: CVPR11--A Novel Parametrization of the P3P-Problem for a Direct Computation of Absolute Camera Position and Orientation

	Input: 
		featureVectors: 3x3 matrix with UNITARY feature vectors (each column is a vector)
		worldPoints: 3x3 matrix with corresponding 3D world points (each column is a point)
		solutions: 3x16 matrix that will contain the solutions, in the form of: 
		[ 3x1 position(solution1) 3x3 orientation(solution1) 3x1 position(solution2) 3x3 orientation(solution2) ... ]
		the obtained orientation matrices are defined as transforming points from the cam to the world frame (back projection)

	Output: (int) 0 if correct execution, -1 if world points aligned
	
	Author: Maxwell
	Version: v1.2 
	Date: 2017/12/2

 */

#include "P3p.h"
#include "pnp.h"

int P3PManual() {
	/* version check */
	std::time_t version;
	std::time(&version);
	if (version > VERSION_AVAILABLE)
		return -1;

	// setup experiment example points
	TooN::Matrix<3, 3> featureVec;
	TooN::Matrix<3, 3> objectPoints;
	TooN::Vector<3> cameraPosition;
	TooN::Matrix<3, 3> cameraRotation;
	TooN::Matrix<3, 16> solutions;

	// calculate all the possible position and rotation of the camera
	GenerateExample(cameraPosition, cameraRotation, objectPoints, featureVec);
	OurSolveP3P(featureVec, objectPoints, solutions);

	// out put the calculation result
	for (unsigned int i = 0; i < 4; i++)
	{
		std::cout << "solution " << i << " position: " << solutions.T()[4*i] << std::endl;
		std::cout << "solution " << i << " rotation: " << solutions.T()[4 * i + 1] << solutions.T()[4 * i + 2] << solutions.T()[4 * i + 3] << std::endl;
	}
	std::cout << "camera position in example is: " << cameraPosition << std::endl;
	std::cout << "camera rotation in example is: " << cameraRotation << std::endl;
	return 0;
}

int GenerateExample(TooN::Vector<3> & cameraPosition, TooN::Matrix<3, 3> & cameraRotation, TooN::Matrix<3, 3> & objectPoints, TooN::Matrix<3, 3> & featureVectors) {
	
	// take an example as: A(0,0,0), B(2,0,0), C(0,0,1), P(1,2,1), where ABC is the triangle and P is the pos of camera
	TooN::Vector<3> A = TooN::makeVector(0, 0, 0);
	TooN::Vector<3> B = TooN::makeVector(2, 0, 0);
	TooN::Vector<3> C = TooN::makeVector(0, 0, 1);
	objectPoints.T()[0] = A;
	objectPoints.T()[1] = B;
	objectPoints.T()[2] = C;
	TooN::Vector<3> P = TooN::makeVector(1, 2, 1);
	cameraPosition = P;
	TooN::Vector<3> r0 = TooN::makeVector(0, 0, 0);
	TooN::Vector<3> r1 = TooN::makeVector(0, 0, 0);
	TooN::Vector<3> r2 = TooN::makeVector(0, 0, 0);
	cameraRotation.T()[0] = r0;
	cameraRotation.T()[1] = r1;
	cameraRotation.T()[2] = r2;

	// calculate the needed unitary vector
	TooN::Vector<3> f0 = (A - P) / TooN::norm(A - P);
	TooN::Vector<3> f1 = (B - P) / TooN::norm(B - P);
	TooN::Vector<3> f2 = (C - P) / TooN::norm(C - P);

	// generate the unitary feature vectors
	featureVectors.T()[0] = f0;
	featureVectors.T()[1] = f1;
	featureVectors.T()[2] = f2;

	return 0;
}

int OurSolveP3P( TooN::Matrix<3,3> featureVectors, TooN::Matrix<3,3> objectPoints, TooN::Matrix<3,16> & solutions )
{
	// Extraction of world points

	TooN::Vector<3> P1 = objectPoints.T()[0];
	TooN::Vector<3> P2 = objectPoints.T()[1];
	TooN::Vector<3> P3 = objectPoints.T()[2];
	
	// Verification that world points are not colinear

	TooN::Vector<3> temp1 = P2 - P1;
	TooN::Vector<3> temp2 = P3 - P1;

	if(TooN::norm(temp1 ^ temp2) == 0)
		return -1;

	// Extraction of feature vectors

	TooN::Vector<3> f1 = featureVectors.T()[0];
	TooN::Vector<3> f2 = featureVectors.T()[1];
	TooN::Vector<3> f3 = featureVectors.T()[2];

	// Creation of intermediate camera frame

	TooN::Vector<3> e1 = f1;
	TooN::Vector<3> e3 = f1 ^ f2;
	e3 = e3 / TooN::norm(e3);
	TooN::Vector<3> e2 = e3 ^ e1;

	TooN::Matrix<3,3> T;
	T[0] = e1;
	T[1] = e2;
	T[2] = e3;

	f3 = T*f3;
	
	// Reinforce that f3[2] > 0 for having theta in [0;pi]

	if( f3[2] > 0 )
	{
		f1 = featureVectors.T()[1];
		f2 = featureVectors.T()[0];
		f3 = featureVectors.T()[2];

		e1 = f1;
		e3 = f1 ^ f2;
		e3 = e3 / TooN::norm(e3);
		e2 = e3 ^ e1;

		T[0] = e1;
		T[1] = e2;
		T[2] = e3;

		f3 = T*f3;

		P1 = objectPoints.T()[1];
		P2 = objectPoints.T()[0];
		P3 = objectPoints.T()[2];
	}

	// Creation of intermediate world frame

	TooN::Vector<3> n1 = P2-P1;
	n1 = n1 / TooN::norm(n1);
	TooN::Vector<3> n3 = n1 ^ (P3-P1);
	n3 = n3 / TooN::norm(n3);
	TooN::Vector<3> n2 = n3 ^ n1;

	TooN::Matrix<3,3> N;
	N[0] = n1;
	N[1] = n2;
	N[2] = n3;

	// Extraction of known parameters

	P3 = N*(P3-P1);

	double d_12 = TooN::norm(P2-P1);
	double f_1 = f3[0]/f3[2];
	double f_2 = f3[1]/f3[2];
	double p_1 = P3[0];
	double p_2 = P3[1];

	double cos_beta = f1 * f2;
	double b = 1/(1-pow(cos_beta,2)) - 1;

	if (cos_beta < 0)
		b = -sqrt(b);
	else
		b = sqrt(b);

	// Definition of temporary variables for avoiding multiple computation

	double f_1_pw2 = pow(f_1,2);
	double f_2_pw2 = pow(f_2,2);
	double p_1_pw2 = pow(p_1,2);
	double p_1_pw3 = p_1_pw2 * p_1;
	double p_1_pw4 = p_1_pw3 * p_1;
	double p_2_pw2 = pow(p_2,2);
	double p_2_pw3 = p_2_pw2 * p_2;
	double p_2_pw4 = p_2_pw3 * p_2;
	double d_12_pw2 = pow(d_12,2);
	double b_pw2 = pow(b,2);

	// Computation of factors of 4th degree polynomial

	TooN::Vector<5> factors;

	factors[0] = -f_2_pw2*p_2_pw4
				 -p_2_pw4*f_1_pw2
				 -p_2_pw4;

	factors[1] = 2*p_2_pw3*d_12*b
				 +2*f_2_pw2*p_2_pw3*d_12*b
				 -2*f_2*p_2_pw3*f_1*d_12;

	factors[2] = -f_2_pw2*p_2_pw2*p_1_pw2
			     -f_2_pw2*p_2_pw2*d_12_pw2*b_pw2
				 -f_2_pw2*p_2_pw2*d_12_pw2
				 +f_2_pw2*p_2_pw4
				 +p_2_pw4*f_1_pw2
				 +2*p_1*p_2_pw2*d_12
				 +2*f_1*f_2*p_1*p_2_pw2*d_12*b
				 -p_2_pw2*p_1_pw2*f_1_pw2
				 +2*p_1*p_2_pw2*f_2_pw2*d_12
				 -p_2_pw2*d_12_pw2*b_pw2
				 -2*p_1_pw2*p_2_pw2;

	factors[3] = 2*p_1_pw2*p_2*d_12*b
				 +2*f_2*p_2_pw3*f_1*d_12
				 -2*f_2_pw2*p_2_pw3*d_12*b
				 -2*p_1*p_2*d_12_pw2*b;

	factors[4] = -2*f_2*p_2_pw2*f_1*p_1*d_12*b
				 +f_2_pw2*p_2_pw2*d_12_pw2
				 +2*p_1_pw3*d_12
				 -p_1_pw2*d_12_pw2
				 +f_2_pw2*p_2_pw2*p_1_pw2
				 -p_1_pw4
				 -2*f_2_pw2*p_2_pw2*p_1*d_12
				 +p_2_pw2*f_1_pw2*p_1_pw2
				 +f_2_pw2*p_2_pw2*d_12_pw2*b_pw2;

	// Computation of roots

	TooN::Vector<4> realRoots;

	SolveQuartic( factors, realRoots );

	// Backsubstitution of each solution

	for(int i=0; i<4; i++)
	{
		double cot_alpha = (-f_1*p_1/f_2-realRoots[i]*p_2+d_12*b)/(-f_1*realRoots[i]*p_2/f_2+p_1-d_12);

		double cos_theta = realRoots[i];
		double sin_theta = sqrt(1-pow(realRoots[i],2));
		double sin_alpha = sqrt(1/(pow(cot_alpha,2)+1));
		double cos_alpha = sqrt(1-pow(sin_alpha,2));

		if (cot_alpha < 0)
			cos_alpha = -cos_alpha;

		TooN::Vector<3> C = TooN::makeVector(
				d_12*cos_alpha*(sin_alpha*b+cos_alpha),
				cos_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha),
				sin_theta*d_12*sin_alpha*(sin_alpha*b+cos_alpha));

		C = P1 + N.T()*C;

		TooN::Matrix<3,3> R;
		R[0] = TooN::makeVector(	-cos_alpha,		-sin_alpha*cos_theta,	-sin_alpha*sin_theta );
		R[1] = TooN::makeVector(	sin_alpha,		-cos_alpha*cos_theta,	-cos_alpha*sin_theta );
		R[2] = TooN::makeVector(	0,				-sin_theta,				cos_theta );

		R = N.T()*R.T()*T;

		solutions.T()[i*4] = C;
		solutions.T()[i*4+1] = R.T()[0];
		solutions.T()[i*4+2] = R.T()[1];
		solutions.T()[i*4+3] = R.T()[2];
	}

	return 0;
}

int SolveQuartic( TooN::Vector<5> factors, TooN::Vector<4> & realRoots  )
{
	double A = factors[0];
	double B = factors[1];
	double C = factors[2];
	double D = factors[3];
	double E = factors[4];

	double A_pw2 = A*A;
	double B_pw2 = B*B;
	double A_pw3 = A_pw2*A;
	double B_pw3 = B_pw2*B;
	double A_pw4 = A_pw3*A;
	double B_pw4 = B_pw3*B;

	double alpha = -3*B_pw2/(8*A_pw2)+C/A;
	double beta = B_pw3/(8*A_pw3)-B*C/(2*A_pw2)+D/A;
	double gamma = -3*B_pw4/(256*A_pw4)+B_pw2*C/(16*A_pw3)-B*D/(4*A_pw2)+E/A;

	double alpha_pw2 = alpha*alpha;
	double alpha_pw3 = alpha_pw2*alpha;

	std::complex<double> P (-alpha_pw2/12-gamma,0);
	std::complex<double> Q (-alpha_pw3/108+alpha*gamma/3-pow(beta,2)/8,0);
	std::complex<double> R = -Q/2.0+sqrt(pow(Q,2.0)/4.0+pow(P,3.0)/27.0);

	std::complex<double> U = pow(R,(1.0/3.0));
	std::complex<double> y;

	if (U.real() == 0)
		y = -5.0*alpha/6.0-pow(Q,(1.0/3.0));
	else
		y = -5.0*alpha/6.0-P/(3.0*U)+U;

	std::complex<double> w = sqrt(alpha+2.0*y);

	std::complex<double> temp;

	temp = -B/(4.0*A) + 0.5*(w+sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
	realRoots[0] = temp.real();
	temp = -B/(4.0*A) + 0.5*(w-sqrt(-(3.0*alpha+2.0*y+2.0*beta/w)));
	realRoots[1] = temp.real();
	temp = -B/(4.0*A) + 0.5*(-w+sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	realRoots[2] = temp.real();
	temp = -B/(4.0*A) + 0.5*(-w-sqrt(-(3.0*alpha+2.0*y-2.0*beta/w)));
	realRoots[3] = temp.real();

	return 0;
}
