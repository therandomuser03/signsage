import { test, expect } from '@playwright/test';

test('landing page has unified navbar', async ({ page }) => {
  await page.goto('/');
  await expect(page).toHaveTitle(/SignSage/);
  await expect(page.getByRole('banner').getByRole('button', { name: 'SignSage' })).toBeVisible();
  await expect(page.getByRole('banner').getByRole('button', { name: 'Translator' }).first()).toBeVisible();
  await expect(page.getByRole('banner').getByRole('button', { name: 'Research' }).first()).toBeVisible();
});

test('translator navigates to workspace', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('banner').getByRole('button', { name: 'Translator' }).first().click();
  await expect(page.getByRole('button', { name: /ISL → English/ })).toBeVisible();
});

test('research navigates to research page', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('banner').getByRole('button', { name: 'Research' }).first().click();
  await expect(page.getByText(/Benchmark/).first()).toBeVisible();
});

test('back via home returns to landing', async ({ page }) => {
  await page.goto('/#/translate/sign');
  await page.getByRole('banner').getByRole('button', { name: 'Home' }).first().click();
  await expect(page.getByText(/Real-Time ISL/)).toBeVisible();
});
