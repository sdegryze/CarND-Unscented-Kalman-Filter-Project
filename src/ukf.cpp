#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  ///* State dimension
  n_x_ = 5;
  
  n_z_ = 3;
  
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // assume a bike rides 20 km/hr == 6 m/s; in 1 sec the bike can probably increase its velocity to 25-30 km/hr
  // so the acceleration standard deviation should somehwere be in the order of 1.5 - 3 m/s^2
  std_a_ = 3.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

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
  
  // Initialize and set the measurement function matrix for laser
  // This is the relation between a laser measurement and the state vector
  H_laser_ = MatrixXd(2, 5);
  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;
  
  // Initialize and set measurement covariance matrix for laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_ * std_laspx_, 0,
              0,                       std_laspy_ * std_laspy_;
  
  
  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;
  
  I_ = MatrixXd::Identity(n_x_, n_x_);
  
  ///* Augmented state dimension
  n_aug_ = 7;
  
  ///* Sigma point spreading parameter
  lambda_ = 3 - n_x_;
  
  ///* Weights of sigma points
  // Note that 2*n_aug_+1 == number of sigma points
  nr_sigma = 2 * n_aug_+1;
  weights_ = VectorXd(nr_sigma);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (int i=1; i < 2 * n_aug_+1; i++) weights_(i) = 1 / (lambda_ + n_aug_) / 2;
  
  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::CreateSigmaPoints(MatrixXd* Xsig_out) {
  //create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;
  
  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;
  
  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();
  
  //create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd(n_aug_, nr_sigma);
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i + 1)       = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
  *Xsig_out = Xsig_aug;
}

void UKF::PredictSigmaPoints(MatrixXd Xsig_aug, float delta_t) {
  MatrixXd Xsig_pred = MatrixXd(n_x_, nr_sigma);
  //predict sigma points
  for (int i = 0; i < nr_sigma; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);
    
    //predicted state values
    double px_p, py_p;
    
    //avoid division by zero
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v / yawd * (sin (yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
    }
    else {
      px_p = p_x + v * delta_t * cos(yaw);
      py_p = p_y + v * delta_t * sin(yaw);
    }
    
    double v_p = v;
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;
    
    //add noise
    px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
    v_p = v_p + nu_a * delta_t;
    
    yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
    yawd_p = yawd_p + nu_yawdd * delta_t;
    
    //write predicted sigma point into right column
    Xsig_pred(0,i) = px_p;
    Xsig_pred(1,i) = py_p;
    Xsig_pred(2,i) = v_p;
    Xsig_pred(3,i) = yaw_p;
    Xsig_pred(4,i) = yawd_p;
  }
  Xsig_pred_ = Xsig_pred;
}

void UKF::CalcMeanAndCovariance(VectorXd* x_out, MatrixXd* P_out) {
  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);
  
  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);
  
  //predicted state mean
  x = Xsig_pred_ * weights_;
  
  //predicted state covariance matrix
  MatrixXd x_diff = Xsig_pred_ - x.replicate(1, nr_sigma);
  tools.NormalizeAngle(&x_diff, 3);
  P = x_diff.cwiseProduct(weights_.transpose().replicate(n_x_, 1)) * x_diff.transpose();

  //write result
  *x_out = x;
  *P_out = P;
}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    
    // Define and set initial state covariance matrix
    MatrixXd P_ = MatrixXd(n_x_, n_x_);
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1, 0, 0,
          0, 0, 0, 1, 0,
          0, 0, 0, 0, 1;

    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      // Convert radar from polar to cartesian coordinates and initialize state
      // assume 0 velocity
      float rho = meas_package.raw_measurements_[0];
      float phi = meas_package.raw_measurements_[1];
      
      float px = rho * cos(phi);
      float py = rho * sin(phi);
      
      // Note that you cannot directly use the radar velocity measurement to initialize the state vector
      // so all other states are set to 0
      x_ << px, py, 0, 0, 0;
      is_initialized_ = true;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      // Initialize state with position of first laser measurement but 0 velocity
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], 0, 0, 0;
      is_initialized_ = true;
    }
    

    if (is_initialized_) {
      cout << "UKF initialized with x_ = " << x_(0) << " " << x_(1) << " " << x_(2) << " " << x_(3) << " " << x_(4) << endl;
    }

    // done initializing, no need to predict or update
    return;
  }
  
  // Predict object position to the current timestep...
  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_package.timestamp_;
  Prediction(delta_t);
  
  // Update the prediction using the new measurement...
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    if (use_radar_) {
      cout << "RADAR with measurements: " << meas_package.raw_measurements_(0) << " " <<
              meas_package.raw_measurements_(1) << " " <<
              meas_package.raw_measurements_(2) << " " << endl;
      float nis = UpdateRadar(meas_package);
      if (radarNisFile != NULL && radarNisFile->is_open()) {
        *radarNisFile << nis << endl;
      }
    }
  } else { // LASER
    if (use_laser_) {
      cout << "LASER with measurements: " << meas_package.raw_measurements_(0) << " " <<
              meas_package.raw_measurements_(1) << " " << endl;
      float nis = UpdateLidar(meas_package);
      if (laserNisFile != NULL && laserNisFile->is_open()) {
        *laserNisFile << nis << endl;
      }
    }

  }
  cout << "x_ = " << x_(0) << " " << x_(1) << " " << x_(2) << " " << x_(3) << " " << x_(4) << endl;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug, P_pred_;
  VectorXd x_pred_;
  CreateSigmaPoints(&Xsig_aug);
  PredictSigmaPoints(Xsig_aug, delta_t);
  CalcMeanAndCovariance(&x_pred_, &P_pred_);
  x_ = x_pred_;
  P_ = P_pred_;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
float UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;
  
  VectorXd z_pred = H_laser_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_laser_.transpose();
  MatrixXd S = H_laser_ * P_ * Ht + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  //new estimate of state vector x_ and covariance matrix P_
  x_ = x_ + (K * y);
  P_ = (I_ - K * H_laser_) * P_;
  
  //calculate and return lidar NIS
  float nis = y.transpose() * Si * y;
  return nis;
}

void UKF::PredictRadarMeasurement(VectorXd z, VectorXd* z_out, MatrixXd* zdiff_out, MatrixXd* S_out) {
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    
    //extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    
    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;
    
    //measurement model
    Zsig(0,i) = sqrt(p_x * p_x + p_y * p_y);                            // r
    Zsig(1,i) = atan2(p_y, p_x);                                        // phi
    Zsig(2,i) = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y);   // r_dot
  }
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  z_pred = Zsig * weights_;
  
  //predicted state covariance matrix
  MatrixXd z_diff = Zsig - z_pred.replicate(1, nr_sigma);
  tools.NormalizeAngle(&z_diff, 1);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z_, n_z_);
  S = z_diff.cwiseProduct(weights_.transpose().replicate(n_z_, 1)) * z_diff.transpose();
  //add measurement noise covariance matrix
  S(0, 0) += std_radr_ * std_radr_;
  S(1, 1) += std_radphi_ * std_radphi_;
  S(2, 2) += std_radrd_ * std_radrd_;
  
  //write result
  *z_out = z_pred;
  *zdiff_out = z_diff;
  *S_out = S;
}

float UKF::UpdateRadarWithPrediction(MatrixXd z_diff, VectorXd z_pred, VectorXd z, MatrixXd S) {
  //calculate cross correlation matrix
  MatrixXd x_diff = Xsig_pred_ - x_.replicate(1, nr_sigma);
  tools.NormalizeAngle(&x_diff, 3);
  MatrixXd Tc;
  Tc = x_diff.cwiseProduct(weights_.transpose().replicate(n_x_, 1)) * z_diff.transpose();
  
  //calculate Kalman gain K;
  MatrixXd Si = S.inverse();
  MatrixXd K = Tc * Si;

  //update state mean x_ and covariance matrix P_
  VectorXd h = z - z_pred;
  tools.NormalizeAngle(&h, 1);
  x_ = x_ + K * h;
  P_ = P_ - K * S * K.transpose();
  
  //calculate and return radar NIS
  float nis = h.transpose() * Si * h;
  return nis;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
float UKF::UpdateRadar(MeasurementPackage meas_package) {
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_pred;
  MatrixXd zdiff, S;
  PredictRadarMeasurement(z, &z_pred, &zdiff, &S);
  float nis = UpdateRadarWithPrediction(zdiff, z_pred, z, S);
  return nis;
}
