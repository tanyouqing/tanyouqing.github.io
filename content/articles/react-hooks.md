---
title: "理解 React Hooks：useState 与 useEffect 深度解析"
date: "2025-01-15"
description: "深入剖析 React Hooks 的内部原理，结合实例讲解 useState、useEffect 的正确使用方式与常见陷阱。"
tags: ["React", "前端开发", "JavaScript"]
category: "技术"
---

## 前言

React Hooks 自 16.8 版本引入以来，彻底改变了我们编写组件的方式。

## useState 的本质

`useState` 本质上是一个闭包，它将状态存储在 React 内部的 Fiber 节点上。

## useEffect 的执行时机

默认情况下，`useEffect` 在每次渲染后执行。通过第二个参数（依赖数组），我们可以精确控制其触发时机。

## 常见陷阱

1. 闭包陷阱：在异步函数中访问过时的状态
2. 无限循环：依赖数组中包含会频繁变化的引用类型
3. 竞态条件：未处理 cleanup 函数导致组件卸载后设置 state