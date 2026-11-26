
# Kinematic Model

## Overview

The main purpose of the kinematic model is to provide differentiable forward kinematics expressions for use in applications such as motion controllers. The model is built using symbolic expressions powered by CasADi, enabling efficient computation and differentiation of kinematic relationships.

## Core Components

### Bodies and Connections

The kinematic model is structured as a tree of connected bodies. Each body represents a physical entity, and connections define the relationships between bodies.

- **Bodies (`Body`)**: Semantic atoms that cannot be decomposed into smaller meaningful parts. They may have visual and collision geometries attached.

- **Connections (`Connection`)**: Define the relative transforms between bodies using 4×4 transformation matrices. There are three main types:
  - **Fixed Connections**: Rigid connections with no degrees of freedom
  - **Active Connections**: Connections with controllable degrees of freedom (e.g., robot joints)
  - **Passive Connections**: Connections with non-fixed but uncontrollable degrees of freedom (e.g., localization)

### Degrees of Freedom

Degrees of freedom (`DegreeOfFreedom`) are variables that can change within specified limits. Each degree of freedom has:

- Position, velocity, acceleration, and jerk components
- Optional upper and lower limits for each derivative
- Symbolic representations for use in kinematic expressions

Active and passive connections use these degrees of freedom to express how bodies can move relative to each other. Connections can share DoFs to create dependent kinematics (e.g., parallel grippers where fingers move together).

### Transformation Matrices

All connections express their relative transforms using 4×4 transformation matrices:
- For fixed connections, matrices contain only constant values
- For active and passive connections, matrix entries are symbolic expressions computed using their DoFs

## Forward Kinematics

Forward kinematics expressions between arbitrary bodies are created by:
1. Searching for a path in the kinematic tree between the bodies
2. Multiplying the transformation matrices of the connections along that path

Key features of the forward kinematics system:
- All expressions are differentiable (useful for control and optimization)
- Expressions can be compiled into fast byte code for efficient computation
- Multiple forward kinematics calculations can be performed simultaneously

## Symbol Management

The `SymbolManager` singleton class manages symbolic variables and their associations with numeric values. It:
- Registers symbols with their value providers
- Resolves symbols to their numeric values during computation
- Handles different mathematical entities (points, vectors, quaternions)
- Evaluates expressions involving these symbols

## World Integration

The kinematic model is integrated into the `World` class, which acts as a mediator for bodies and connections. The world:
- Manages the collection of bodies, connections, and degrees of freedom
- Handles validation of the kinematic structure
- Provides methods to find paths between bodies
- Computes forward kinematics between arbitrary bodies
- Can merge other worlds into itself, combining their kinematic structures

## Example Usage

Common operations with the kinematic model include:
- Creating bodies and connections to build kinematic chains
- Setting up degrees of freedom with appropriate limits
- Computing forward kinematics between arbitrary bodies
- Compiling expressions for efficient repeated computation
- Using the differentiable nature of the expressions for control or optimization

## Citation
The world model is also explained in chapter 4 of {cite:t}`stelter25giskard`.