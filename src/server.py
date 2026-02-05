#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Description Server Entry Point
  @Author Chris
  @Date 2026/2/2
"""
import model.corrector
import model.inferencer
import model.trainer

N = 1


def main():
    for _ in range(N):
        model.inferencer.main()
        model.corrector.main()


if __name__ == "__main__":
    main()
