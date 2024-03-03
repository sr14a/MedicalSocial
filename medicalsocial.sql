-- phpMyAdmin SQL Dump
-- version 4.0.4
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Nov 05, 2023 at 08:33 AM
-- Server version: 5.6.12-log
-- PHP Version: 5.4.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `medicalsocial`
--
CREATE DATABASE IF NOT EXISTS `medicalsocial` DEFAULT CHARACTER SET latin1 COLLATE latin1_swedish_ci;
USE `medicalsocial`;

-- --------------------------------------------------------

--
-- Table structure for table `comment`
--

CREATE TABLE IF NOT EXISTS `comment` (
  `post_id` varchar(30) DEFAULT NULL,
  `username` varchar(30) DEFAULT NULL,
  `comment` varchar(200) DEFAULT NULL,
  `rate` varchar(10) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `comment`
--

INSERT INTO `comment` (`post_id`, `username`, `comment`, `rate`) VALUES
('1', 'abhinay', 'good', '5'),
('2', 'abhinay', 'bad', '1'),
('3', 'Raju', 'good', '3');

-- --------------------------------------------------------

--
-- Table structure for table `post`
--

CREATE TABLE IF NOT EXISTS `post` (
  `username` varchar(30) DEFAULT NULL,
  `post_id` varchar(50) DEFAULT NULL,
  `image` varchar(100) DEFAULT NULL,
  `name` varchar(100) DEFAULT NULL,
  `topic` varchar(100) DEFAULT NULL,
  `description` varchar(100) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `post`
--

INSERT INTO `post` (`username`, `post_id`, `image`, `name`, `topic`, `description`) VALUES
('abhinay', '1', 'men.png', 'abhinay', 'happy', 'happy is my gole'),
('abhinay', '2', 'men.png', 'abhinay', 'sad', 'sad is my gole'),
('Raju', '3', 'men.png', 'Raju', 'good fruit', 'happy is my gole');

-- --------------------------------------------------------

--
-- Table structure for table `register`
--

CREATE TABLE IF NOT EXISTS `register` (
  `username` varchar(30) NOT NULL,
  `password` varchar(30) DEFAULT NULL,
  `contact` varchar(12) DEFAULT NULL,
  `email` varchar(30) DEFAULT NULL,
  `address` varchar(40) DEFAULT NULL,
  `status` varchar(200) DEFAULT NULL,
  PRIMARY KEY (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `register`
--

INSERT INTO `register` (`username`, `password`, `contact`, `email`, `address`, `status`) VALUES
('abhinay', '123456', '9874563210', 'abhinay0703@gmail.com', 'Hyderabad', 'good'),
('Raju', '123456', '9874563211', 'raju@gmail.com', 'Hyderabad', 'good');

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
