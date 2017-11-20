#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2 (Tunable)
  std_a_ = 1.3;

  // Process noise standard deviation yaw acceleration in rad/s^2 (Tunable)
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */

  // Number of state variables
  n_x_ = 5;

  // Number of augmented variables
  n_aug_ = 7;

  // number of sigma points
  n_sig_ = 2 * n_aug_ + 1;

  // It is not yet initialized
  is_initialized_ = false;

  // Compute the lambda value (spreading parameter)
  lambda_ = 3 - n_aug_;

  // Initialize the state
  x_.setOnes();

  // Initialize covariance matrix as an identity matrix
  P_.setIdentity();

  // create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  Xsig_pred_.fill(0.0);

  // create vector for weights
  weights_ = VectorXd(n_sig_);
  
  // initialize the weights
  weights_.fill(0.5 /  (lambda_ + n_aug_));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  // measurement noise from laser
  R_LASER_ = MatrixXd(2, 2);
  R_LASER_.setZero();
  R_LASER_.diagonal() << pow(std_laspx_, 2), pow(std_laspy_, 2);

  // measurement noise from radar
  R_RADAR_ = MatrixXd(3, 3);
  R_RADAR_.setZero();
  R_RADAR_.diagonal() << pow(std_radr_, 2), pow(std_radphi_, 2), pow(std_radrd_, 2);
}
 
UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */

  float px, py, v, psi, psi_dot = 0;

  // if its the first time
  if (!is_initialized_) {
    cout << "UKF: " << endl;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_[0];
      float theta = meas_package.raw_measurements_[1];
      float ro_dot = meas_package.raw_measurements_[2];

      // convert from polar to cartesian coordinates
      px = ro * cos(theta);
      py = ro * sin(theta);

      v = ro_dot;
      psi = theta;
      psi_dot = 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      px = meas_package.raw_measurements_[0];
      py = meas_package.raw_measurements_[1];

      v = 0;
      psi = 0;
      psi_dot = 0;
    }

    // close to zero check
    if (fabs(px) < 0.0001) {
      px = 0.01;
      cout << "init px too small" << endl;
    }

    if (fabs(py) < 0.0001) {
      py = 0.01;
      cout << "init py too small" << endl;
    }
    
    // set the values to the state
    x_ << px, py, v, psi, psi_dot;

    // update the timestamp
    time_us_ = meas_package.timestamp_;
    
    // update the flag that we have initialized state and co-variance matrix
    is_initialized_ = true;

    cout << "Initialized" << endl;
    return;
  }


  float delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  // update the timestamp
  time_us_ = meas_package.timestamp_;
    
  // Predict
  Prediction(delta_t);
    
  // Update
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }
  
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */ 

  ////////////////////////////////////
  // Step:1 Generate the sigma points
  ///////////////////////////////////

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  Xsig_aug.fill(0.0);

  // create augmented mean state (Noises have a mean of zero so we keep them as 0)
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  // create augmented covariance matrix
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_ + 1) = std_a_ * std_a_;

  // create square root matrix (used in the calculation)
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; i++) 
  {
    Xsig_aug.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(n_aug_ + i + 1) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  }

  ////////////////////////////////////
  // Step:2 Translate the points
  ///////////////////////////////////
  
  //predict sigma points
  for (int i = 0; i < n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin(yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd*delta_t));
    }
    else {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;
  }

  ////////////////////////////////////
  // Step:3 Predict mean and covariance
  ///////////////////////////////////

  // create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);

  // create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);

  // predict state mean
  MatrixXd w = weights_.transpose();
  MatrixXd sig = Xsig_pred_.transpose();
  MatrixXd out = w * sig;
  x = out.transpose();

  // predict state covariance matrix  
  for (int i = 0; i < n_sig_; i++) {
    MatrixXd variance = Xsig_pred_.col(i) - x;
    
    //angle normalization
    while (variance(3)> M_PI) variance(3) -= 2.*M_PI;
    while (variance(3)<-M_PI) variance(3) += 2.*M_PI;

    P = P + weights_(i) * variance * variance.transpose();
  }

  // update the state and co-variance with predicted values
  x_ = x;
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  ////////////////////////////////////
  // Step:1 Predict the new state
  ///////////////////////////////////

  // set measurement dimension, lidar can measure px, py, and r_dot
  int n_z = 2;

  //set vector for weights
  double weight_0 = lambda_ / (lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < n_sig_; i++) {
    double weight = 0.5 / (n_aug_ + lambda_);
    weights_(i) = weight;
  }

  // create matrix for sigma points in measurement space  
  MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // calculate mean predicted measurement
  z_pred = weights_.transpose() * Zsig.transpose();

  // calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  S = S + R_LASER_;

  //////////////////////////////////////
  //// Step:2 Update using measurement
  /////////////////////////////////////
  float px = meas_package.raw_measurements_[0];
  float py = meas_package.raw_measurements_[1];  

  VectorXd z = VectorXd(n_z);
  z << px, py;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 1 + 2 * n_aug_; i++) {
    MatrixXd x_diff = Xsig_pred_.col(i) - x_;
    MatrixXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) <-M_PI) x_diff(3) += 2.*M_PI;

    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;

    Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // normalize the angle
  z_diff(1) = Tools::NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */

  ////////////////////////////////////
  // Step:1 Predict the new state
  ///////////////////////////////////

  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);

  // transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {
    double px = Xsig_pred_(0, i);
    double py = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);

    // close to zero check
    if (fabs(px) < 0.0001) {
      px = 0.01;
      cout << "init px too small" << endl;
    }

    if (fabs(py) < 0.0001) {
      py = 0.01;
      cout << "init py too small" << endl;
    }

    double radr = sqrt(px*px + py*py);
    double radphi = atan2(py, px);
    double radrd = (px * cos(psi) * v + py * sin(psi) * v) / (radr);

    Zsig.col(i) << radr, radphi, radrd;
  }

  // calculate mean predicted measurement
  z_pred = weights_.transpose() * Zsig.transpose();

  // calculate measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {
    MatrixXd z_diff = Zsig.col(i) - z_pred;

    // angle normalization
    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }
  
  S = S + R_RADAR_;

  ////////////////////////////////////
  // Step:2 Update using measurement
  ///////////////////////////////////
  float ro = meas_package.raw_measurements_[0];
  float theta = meas_package.raw_measurements_[1];
  float ro_dot = meas_package.raw_measurements_[2];

  VectorXd z = VectorXd(n_z);
  z << ro, theta, ro_dot;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < 1 + 2 * n_aug_; i++) {
    MatrixXd x_diff = Xsig_pred_.col(i) - x_;
    MatrixXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
    while (x_diff(3) <-M_PI) x_diff(3) += 2.*M_PI;

    while (z_diff(1) > M_PI) z_diff(1) -= 2.*M_PI;
    while (z_diff(1) <-M_PI) z_diff(1) += 2.*M_PI;

    Tc = Tc + weights_[i] * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd K = Tc * S.inverse();

  //update state mean and covariance matrix
  VectorXd z_diff = z - z_pred;

  // normalize the angle
  z_diff(1) = Tools::NormalizeAngle(z_diff(1));

  x_ = x_ + K * z_diff;
  P_ = P_ - K * S * K.transpose();


}
