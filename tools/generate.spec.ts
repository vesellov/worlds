import { test, expect } from '@playwright/test'

test.describe('map layers', () => {
  test.beforeEach(async ({ context, page }) => {
    await context.clearCookies()
    await page.goto('/')
    await page.evaluate(() => {
      localStorage.clear()
      sessionStorage.clear()
    })
    await page.goto('/?width=1024&height=1024&options=default')  // &&seed=1
    await page.waitForFunction(() => (window as any).mapId !== undefined, { timeout: 60000 })
    await page.waitForTimeout(500)
  })

  test('generate map', async ({ page }) => {
    await page.click('#optionsTrigger')
    await page.click('#optionsTab')

    const culturesOutput = page.locator('#culturesOutput')
    await page.locator('#culturesOutput').fill('3')
    const status = await page.evaluate(async () => {
      document.getElementById('statesNumber').children.item(1).value = "6";
      document.getElementById('lock_statesNumber').setAttribute('data-locked','1');
    })

    await page.locator('#newMapButton').click()

    const exportButton = page.locator('#exportButton')
    await page.click('#exportButton')
    const exportToJsonFull = page.getByText('full', { exact: true })
    const downloadPromise = page.waitForEvent('download')
    await page.getByText('full', { exact: true }).click()

    const download = await downloadPromise
    await download.saveAs('map.json')

    // await page.waitForTimeout(10000)

  })

})
