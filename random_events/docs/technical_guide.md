# Technical Guide

This document guides through the structure of the `random_events` package and explains how it works.

The package is a python wrapper around a C++ library working in the same way, hence the technical documentation here
also applies to the C++ backend.

The most important classes are the `AbstractSimpleSet` and `AbstractCompositeSet` classes.

```{eval-rst}
   .. inheritance-diagram::  random_events.product_algebra random_events.interval.SimpleInterval random_events.interval.Interval random_events.set random_events.variable
      :parts: 2
```

You can open the C++ project by using CLion.
For this you need the bazel plugin for CLion and import the submodule in `src/random-events-lib`.
Next, open the existing bazel file and continue the process from there. This gets you set up to run the C++ tests.

## New Algebras

Algebraic structures are the main concern of this package.
When you want to add a new algebra, you need to create a new class that inherits from
`AbstractSimpleSet` and `AbstractCompositeSet`.

`SimpleInterval` and `Interval` provide an easy to grasp example of such an implementation.

## New Variables

Describing a multidimensional algebra needs variables that describe the type of dimensions.
In more applied scenarios, it may happen that more information is needed to describe a variable.
Achieving this requires the creation of a new class that inherits from the respective variable.

## Serialization

The `SubClassJSONSerializer` is a class that automatically serializes and deserializes subclasses of a said class.
I strongly recommended getting familiar with this class when adding new classes to the package.

## Association over Inheritance

Always be aware that inheritance is defined as

> What is wanted here is something like the following substitution property: If for each object o1 of type S there is an
> object o2 of type T such that for all programs P defined in terms of T, the behavior of P is unchanged when o1 is
> substituted for o2 then S is a subtype of T.

So if you are thinking of extending the algebra or variables, be very sure that what you want is really inheritance and
not association.