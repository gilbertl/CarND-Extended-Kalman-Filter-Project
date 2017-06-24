#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truths) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  //accumulate squared residuals
  int n = 0;
  for(int i=0; i < estimations.size(); ++i){
      VectorXd estimation = estimations[i];
      VectorXd ground_truth = ground_truths[i];
      if (estimation.size() == 0) {
          continue;
      }
      if (ground_truth.size() != estimation.size()) {
          continue;
      }
      VectorXd diff = estimation - ground_truth;
      VectorXd diff2 = diff.array() * diff.array();
      rmse += diff2;
      n += 1;
  }
  rmse /= n;
  rmse = rmse.array().sqrt();

  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float c1 = px*px+py*py;
  float c2 = sqrt(c1);
  float c3 = (c1*c2);

  //check division by zero
  if(fabs(c1) < 0.0001){
      cout << "CalculateJacobian () - Error - Division by Zero" << endl;
      return Hj;
  }

  //compute the Jacobian matrix
  Hj << (px/c2), (py/c2), 0, 0,
        -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

  return Hj;
}
