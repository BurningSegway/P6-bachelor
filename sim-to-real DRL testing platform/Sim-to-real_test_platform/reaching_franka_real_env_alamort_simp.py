import gym
import time
import threading
import numpy as np
from packaging import version

import frankx


class ReachingFranka(gym.Env):
    def __init__(self, robot_ip="172.16.0.2", device="cuda:0", control_space="joint", motion_type="impedance", camera_tracking=False):
        # gym API
        self._drepecated_api = version.parse(gym.__version__) < version.parse(" 0.25.0")

        self.device = device
        self.control_space = control_space  # joint or cartesian
        self.motion_type = motion_type  # waypoint or impedance
        
        if self.control_space == "cartesian" and self.motion_type == "impedance":
            # The operation of this mode (Cartesian-impedance) was adjusted later without being able to test it on the real robot.
            # Dangerous movements may occur for the operator and the robot.
            # Comment the following line of code if you want to proceed with this mode.
            raise ValueError("See comment in the code to proceed with this mode")
            pass

        # camera tracking (disabled by default)
        self.camera_tracking = camera_tracking
        if self.camera_tracking:
            threading.Thread(target=self._update_target_from_camera).start()

        # spaces
        self.observation_space = gym.spaces.Box(low=-1000, high=1000, shape=(19,), dtype=np.float32)



        if self.control_space == "joint": #setting the control space as the blind agents
            #self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32) #setting the action space (7 for joint and 1 for gripper)
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)
        else:
            raise ValueError("Invalid control space:", self.control_space)
        
        # init real franka
        print("Connecting to robot at {}...".format(robot_ip))
        self.robot = frankx.Robot(robot_ip)
        self.gripper = frankx.Gripper(robot_ip)  # gwippa init
        self.robot.set_default_behavior()
        self.robot.recover_from_errors()

        # the robot's response can be better managed by independently setting the following properties, for example:
        self.robot.velocity_rel = 0.15
        self.robot.acceleration_rel = 0.05
        self.robot.jerk_rel = 0.005
        #self.robot.set_dynamic_rel(0.2)

        state = self.robot.read_once()
        #print('\nPose: ', self.robot.current_pose())
        #print('O_TT_E: ', state.O_T_EE)

        self.motion = None
        self.motion_thread = None

        #setting parameters
        self.dt = 1 / 120
        self.action_scale = 2.5
        self.dof_vel_scale = 0.1
        self.max_episode_length = 200
        self.robot_dof_speed_scales = 1 #This controlls the speed do not put above 0.5
        self.robot_default_dof_pos = np.array([-0.217, 0.698, 0.050, 0.239, 0.435, 0.767, -1.175])
        #self.robot_default_dof_pos = np.radians([0, -45, 0, -135, 0, 90, 45])
        self.robot_dof_lower_limits = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.robot_dof_upper_limits = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
        self.progress_buf = 1
        self.obs_buf = np.zeros((19,), dtype=np.float32)


    def _update_target_from_camera(self):
        pixel_to_meter = 1.11 / 375  # m/px: adjust for custom cases

        import cv2
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # convert to HSV and remove noise
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv = cv2.medianBlur(hsv, 15)

            # color matching in HSV
            mask = cv2.inRange(hsv, np.array([80, 100, 100]), np.array([100, 255, 255]))
            M = cv2.moments(mask)
            if M["m00"]:
                x = M["m10"] / M["m00"]
                y = M["m01"] / M["m00"]

                # real-world position (fixed z to 0.2 meters)
                pos = np.array([pixel_to_meter * (y - 185), pixel_to_meter * (x - 320), 0.2])
                if self is not None:
                    self.target_pos = pos

                # draw target
                frame = cv2.circle(frame, (int(x), int(y)), 30, (0,0,255), 2)
                frame = cv2.putText(frame, str(np.round(pos, 4).tolist()), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

            # show images
            cv2.imshow("frame", frame)
            cv2.imshow("mask", mask)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                cap.release()

    def _get_observation_reward_done(self):
        # get robot state
        try:
            robot_state = self.robot.get_state(read_once=True)
        except frankx.InvalidOperationException:
            robot_state = self.robot.get_state(read_once=False)

        #self.gripper = self.robot.get_gripper()
        gripper_width = self.gripper.width()

        self.joint_pos = np.zeros((9,))
        self.joint_pos[0:7] = np.array(robot_state.q)
        self.joint_pos[7:8] = gripper_width/2
        self.joint_pos[8:9] = gripper_width/2

        self.joint_vel = np.zeros((7,))
        self.joint_vel[0:7] = np.array(robot_state.dq)

        self.object_position = self.target_pos
   
        self.obs_buf[0:9] =  self.joint_pos
        self.obs_buf[9:16] = self.joint_vel
        self.obs_buf[16:19] = self.object_position

        # get robot state

        # reward
        #distance = np.linalg.norm(end_effector_pos - self.target_pos)
        #reward = -distance
        reward = 0

        # done
        done = self.progress_buf >= self.max_episode_length - 1
        done = done# or distance <= 0.075

        #print("Distance:", distance)
        if done:
            print("Target or Maximum episode length reached")
            time.sleep(1)

        return self.obs_buf, reward, done


    def reset(self):
        print("Resetting...")

        # end current motion
        if self.motion is not None:
            self.motion.finish()
            self.motion_thread.join()
        self.motion = None
        self.motion_thread = None
        

        self.gripper.move_async(0.08)

        # go to 1) safe position, 2) random position
        self.robot.move(frankx.JointMotion(self.robot_default_dof_pos.tolist()))
        dof_pos = self.robot_default_dof_pos # + 0.25 * (np.random.rand(7) - 0.5) #start position offset
        self.robot.move(frankx.JointMotion(dof_pos.tolist()))


        # get target position from prompt
        if not self.camera_tracking:
            while True:
                try:
                    print("Enter target position (X, Y, Z) in meters")
                    raw = input("or press [Enter] key for a random target position: ")
                    if raw:
                        self.target_pos = np.array([float(p) for p in raw.replace(' ', '').split(',')])
                    else:
                        #################################################################
                        #self.target_pos = np.array([0.6, 0.0, 0.0]) #!!Where is the rock!!
                        #################################################################

                        #Række 1
                        self.target_pos = np.array([0.65, 0.34, 0.0])
                        #self.target_pos = np.array([0.65, 0.17, 0.0])
                        #self.target_pos = np.array([0.65, 0.0, 0.0])
                        #self.target_pos = np.array([0.65, -0.17, 0.0])
                        #self.target_pos = np.array([0.65, -0.34, 0.0])
                        #Række 2
                        #self.target_pos = np.array([0.51, 0.34, 0.0])
                        #self.target_pos = np.array([0.51, 0.17, 0.0])
                        #self.target_pos = np.array([0.51, 0.0, 0.0])
                        #self.target_pos = np.array([0.51, -0.17, 0.0])
                        #self.target_pos = np.array([0.51, -0.34, 0.0])
                        #Række 3
                        #self.target_pos = np.array([0.35, 0.34, 0.0])
                        #self.target_pos = np.array([0.35, 0.17, 0.0])
                        #self.target_pos = np.array([0.35, 0.0, 0.0])
                        #self.target_pos = np.array([0.35, -0.17, 0.0])
                        #self.target_pos = np.array([0.35, -0.34, 0.0])
                        #Række 4
                        #self.target_pos = np.array([0.24, 0.34, 0.0])
                        #self.target_pos = np.array([0.24, 0.17, 0.0])
                        #self.target_pos = np.array([0.24, -0.17, 0.0])
                        #self.target_pos = np.array([0.24, -0.34, 0.0])
                        #Række 5
                        #self.target_pos = np.array([0.1, 0.34, 0.0])
                        #self.target_pos = np.array([0.1, -0.34, 0.0])






                        #Within area
                        #Række 1
                        #self.target_pos = np.array([0.6, 0.25, 0.0])
                        #self.target_pos = np.array([0.6, 0.08, 0.0])
                        #self.target_pos = np.array([0.6, -0.08, 0.0])
                        #self.target_pos = np.array([0.6, -0.25, 0.0])
                        #Række 2
                        #self.target_pos = np.array([0.53, 0.25, 0.0])
                        #self.target_pos = np.array([0.53, 0.08, 0.0])
                        #self.target_pos = np.array([0.53, -0.08, 0.0])
                        #self.target_pos = np.array([0.53, -0.25, 0.0])
                        #Række 3
                        #self.target_pos = np.array([0.46, 0.25, 0.0])
                        #self.target_pos = np.array([0.46, 0.08, 0.0])
                        #self.target_pos = np.array([0.46, -0.08, 0.0])
                        #self.target_pos = np.array([0.46, -0.25, 0.0])
                        #Række 4
                        #self.target_pos = np.array([0.4, 0.25, 0.0])
                        #self.target_pos = np.array([0.4, 0.08, 0.0])
                        #self.target_pos = np.array([0.4, -0.08, 0.0])
                        #self.target_pos = np.array([0.4, -0.25, 0.0])

                        #Tilfældige


                    print("Target position:", self.target_pos)
                    break
                except ValueError:
                    print("Invalid input.")

        # initial pose with gripper
        affine = frankx.Affine(frankx.Kinematics.forward(dof_pos.tolist()))
        affine = affine * frankx.Affine(x=0, y=0, z=-0.10335, a=np.pi/2)
        
        # motion type
        if self.motion_type == "waypoint":
            self.motion = frankx.WaypointMotion([frankx.Waypoint(affine)], return_when_finished=False)
        elif self.motion_type == "impedance":
            self.motion = frankx.ImpedanceMotion(500, 50)
        else:
            raise ValueError("Invalid motion type:", self.motion_type)

        self.motion_thread = self.robot.move_async(self.motion)
        if self.motion_type == "impedance":
            self.motion.target = affine

        input("Press [Enter] to continue")

        self.progress_buf = 0
        observation, reward, done = self._get_observation_reward_done()

        if self._drepecated_api:
            return observation
        else:
            return observation, {}
    
    def step(self, action):

        self.progress_buf += 1
        # control space
        
    ###Dof action scaleing to avoid ilegal comands
        def translate_dof_a_to_c(values, original_min, original_max, pi):
            left_span = 40
            right_span = 2 * pi  # New range is [-py, py], so total span is 2*py
    
            # Scale to [0, 1] first, then to [-py, py]
            normalized = (values - original_min) / left_span  # [0, 1]
            dof_c = normalized * right_span - pi  # [-py, py]
    
            return dof_c
    ### 
        #scaled_dof = translate_dof_a_to_c(action[0:7], -20, 20, np.pi)
        #action[0:7] = translate_dof_a_to_c(action[0:7], -20, 20, np.pi)
        #scaled_dof = (action[0:7])
        # joint
        if self.control_space == "joint":
            # get robot state
            try:
                robot_state = self.robot.get_state(read_once=True)
            except frankx.InvalidOperationException:
                robot_state = self.robot.get_state(read_once=False)
            
            # forward kinematics
            dof_pos = np.array(robot_state.q) + (self.robot_dof_speed_scales * self.dt * action[0:7] * self.action_scale)
            #dof_pos = np.array(robot_state.q) + (self.robot_dof_speed_scales * self.dt * scaled_dof * self.action_scale)
            affine = frankx.Affine(self.robot.forward_kinematics(dof_pos.flatten().tolist()))
            
            affine = affine * frankx.Affine(x=0, y=0, z=-0.10335, a=np.pi/2) # adds end effector transform so affine base to end effector transform
        

        # impedance motion
        ###
                    #self.gripper.move_async(action[7:8])#gripper comand here?
        
        #Gripper action goes here
        #This code makes the gripper action executeble (maps from 20 :  to 0.08 : 0)
        def translate_a_to_c(value, leftMin, leftMax, rightMin, rightMax):
            # Figure out how 'wide' each range is
            leftSpan = leftMax - leftMin
            rightSpan = rightMax - rightMin
            if value >= leftMax:
                value == leftMax
            # Convert the left range into a 0-1 range (float)
            valueScaled = float(value - leftMin) / float(leftSpan)

            # Convert the 0-1 range into a value in the right range.
            return rightMin + (valueScaled * rightSpan)

        #def translate_c_to_a(value, rightMin, rightMax, leftMin, leftMax):
        #    # Figure out how 'wide' each range is
        #    leftSpan = leftMax - leftMin
        #    rightSpan = rightMax - rightMin
        #
        #    # Convert the left range into a 0-1 range (float)
        #    valueScaled = float(value - rightMin) / float(rightSpan)
        #
        #    # Convert the 0-1 range into a value in the right range.
        #    return leftMin + (valueScaled * leftSpan)

        #gripper_action = (action[7:8]*0.01)/2
        
        #gripper_width = self.gripper.width()
        #gripper_width_target = (translate_a_to_c(action[7:8], -10, 10, 0.00, 0.08))
        #self.gripper.move_async(gripper_width_target)
        #print("Gripper width target is:",gripper_width_target)

        # the use of time.sleep is for simplicity. This does not guarantee control at a specific frequency
        time.sleep(0.1)  # lower frequency, at 30Hz there are discontinuities

        observation, reward, done = self._get_observation_reward_done()
        
        #print("DEBUGGING")
        #print("action is: ", action)
        #print("Affine is: ", affine)
        #obs_array_type = ["joint_pose", "gripper_pose", "joint_vel", "object_pose"]
        #for i in range(len(observation)):
        #    if i<=6: print(obs_array_type[0], i+1, "is: ", observation[i])
        #    if 6<i<=8: print(obs_array_type[1], i+7, "is: ", observation[i])
        #    if 8<i<=15: print(obs_array_type[2], "is: ", observation[i])
        #    if 15<i<19: print(obs_array_type[2], "is: ", observation[i])

        if self._drepecated_api:
            return observation, reward, done, {}
        else:
            return observation, reward, done, done, {}

    def render(self, *args, **kwargs):
        pass

    def close(self):
        
        pass
